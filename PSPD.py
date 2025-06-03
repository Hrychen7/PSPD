# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:55:55 2024

@author: ChenPiQ
"""
from sys import prefix
import os
import torch
import logging
import random
import numpy as np
import warnings
import math
import statistics
from args import get_parser
from dataset.data import data_prefetcher, AllData_DataFrame
import torch.distributed as dist
from models.resnetV2 import resnet18, resnet101, resnet50
# from apex import amp
import torch.nn.functional as F
# from apex.parallel import DistributedDataParallel
from torch.nn.parallel import DistributedDataParallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import models.context
import torch.nn as nn
from timm.utils import NativeScaler
from utils.utils import reduce_mean,adjust_learning_rate, AverageMeter, ProgressMeter, my_KLDivLoss
root = "./"


def initialize():
    # get args
    args = get_parser()

    # warnings
    warnings.filterwarnings("ignore")

    # logger
    logger = logging.getLogger(__name__)

    # set seed
    seed = int(1111)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    # initialize logger
    logger.setLevel(level = logging.INFO)

    if not os.path.exists(os.path.join(root,"logs")):
        os.makedirs(os.path.join(root,"logs"))

    handler = logging.FileHandler(os.path.join(root,"logs/%s.txt" % args.env_name))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)
    return args, logger

def main():
    config, logger = initialize()
    config.nprocs = torch.cuda.device_count()
    main_worker(config, logger)
    
def curr_v(l, lamda, spl_type='hard'):
    if spl_type == 0: # hard
        v = (l < lamda).float()
        g = -lamda * (v.sum())
    elif spl_type == 1: # soft
        v = (l < lamda).float()
        v *= (1 - l / lamda)
        g = 0.5 * lamda * (v * v - 2 * v).sum()
    elif spl_type == 2: # log
        v = (1 + math.exp(-lamda)) / (1 + (l - lamda).exp())
        mu = 1 + math.exp(-lamda) - v
        g = (mu * mu.log() + v * (v+1e-8).log() - lamda * v)
        # print(g, v.min(), v)

    else:
        raise NotImplementedError('Not implemented of spl type {}'.format(spl_type))
    
    return v
def lambda_scheduler(lambda_0, iter, alpha=0.0001, iter_0=100):
    if iter < iter_0:
        lamda = lambda_0 + alpha * iter
    else:
        lamda = lambda_0 + alpha * iter_0
    return lamda
def weight_funce(func,x,y,lamda,spl_type):
    n=x.shape[0]
    ret = 0.
    for i in range(n):
        m=curr_v(func(x[i][None],y[i][None]),lamda,spl_type)
        ret += func(x[i][None],y[i][None]) *m
       
    return ret / n

def weight_funkd(func1,x1,y1,func2,x2,y2,lamda,spl_type):
    n=x1.shape[0]
    ret = 0.
    for i in range(n):
        m=curr_v(func2(x2[i][None],y2[i][None]),lamda,spl_type)
        ret += func1(x1[i],y1[i]) *m
    return ret / n

def main_worker(config, logger):
    model_names = ["resnet18", "resnet50","resnet101" ]
    models = [resnet18, resnet50, resnet101]

    best_acc1 = -99.0
    best_acc2 = -99.0
    best_auc = -99.0
    dist.init_process_group(backend='nccl')
    # create model
    model_t = models[model_names.index(config.arch)](output_dim=2, mode = config.mode)
    model_s = models[model_names.index(config.arch)](output_dim=2, mode = config.mode)
    
    class PSKD(nn.Module):
        def __init__(self, teacher, student):
            super(PSKD, self).__init__()
            self.teacher = teacher
            self.student = student

        def load_previous_time(self,pth):
            self.teacher.load_state_dict(pth)

        def forward_teacher(self,x, T):
            return self.teacher(x, T)

        def forward_student(self,x, T):
            return self.student(x,T)

        def forward(self,x,T, is_train = True):
            if is_train is True:
                return self.teacher(x, T), self.student(x, T)
            else:
                return self.student(x, T)

        def to_train(self):
            self.teacher.eval()
            self.student.train()

        def to_eval(self):
            self.teacher.eval()
            self.student.eval()

        @torch.no_grad()
        def update_target(self):
            for param_t, param_s in zip(self.teacher.parameters(), self.student.parameters()):
                param_t.data = param_s.data #param_k.data * self.m + param_q.data * (1. - self.m)

    all_model = PSKD(model_t, model_s)
    torch.cuda.set_device(config.local_rank)
    all_model.cuda().eval()

    logger.info("Loading finished.")
    config.batch_size = int(config.batch_size / config.nprocs)
    
    # all_model = DistributedDataParallel(all_model, find_unused_parameters=False)
    all_model = DistributedDataParallel(all_model, find_unused_parameters=True)
    all_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(all_model)
    
    optimizer = torch.optim.AdamW(all_model.module.student.parameters(),lr = config.lr,weight_decay = 0.0005)
    loss_scaler = NativeScaler()

    # all_model, optimizer = amp.initialize(all_model, optimizer, opt_level=config.opt_level)
    
    cudnn.benchmark = True

    # Data loading code
    train_data = AllData_DataFrame("/mnt/sdb/chr/PSPD/pace_dataset.csv",config, train = True)
    val_data = AllData_DataFrame("/mnt/sdb/chr/PSPD/pace_dataset.csv",config, train = False)
    test_data = AllData_DataFrame("/mnt/sdb/chr/PSPD/pace_dataset.csv", config,train = False, test = True)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)

    train_loader = DataLoader(train_data,config.batch_size,
                        shuffle=False,num_workers=8,pin_memory = True, sampler = train_sampler)
    val_loader = DataLoader(val_data,config.batch_size,
                        shuffle=False,num_workers=8,pin_memory = False, sampler = val_sampler)
    test_loader = DataLoader(test_data,config.batch_size,
                       shuffle=False,num_workers=8,pin_memory = False, sampler = test_sampler)
    print(config.ce_weight)
    check_path = os.path.join(root,"checkpoints/mine/%s" % config.env_name)
    best_test_acc  = -1
    save_lbl = 0
    save_pred = 0
    for epoch in range(config.epochs):
        train_sampler.set_epoch(epoch)

        # adjust_learning_rate(optimizer, epoch, config)
        import time
        # train for one epoch
        st_time = time.time()
        train(train_loader, all_model,loss_scaler, optimizer, epoch, config, logger)
        end_time = time.time()
        run_time = end_time - st_time
        logger.info(f"Running time for epoch: {run_time}")
        acc,auc,_,_ = validate(val_loader,all_model, config, logger, prefix = "val")

        is_best = (acc > best_acc1) or (acc == best_acc1 and auc > best_auc)
        if is_best:
            test_acc,_, test_lbl, test_pred = validate(test_loader,all_model, config, logger, prefix = "test")
            best_test_acc = test_acc
            save_lbl = test_lbl
            save_pred = test_pred
        print("best acc: ", best_test_acc)
        best_acc1 = max(acc, best_acc1)
        best_auc = max(auc, best_auc)
        
        if not os.path.exists(check_path):
            try:
                os.makedirs(check_path)
            except:
                pass # multiple processors bug
        state = {
                'epoch': epoch + 1,
                'state_dict': all_model.module.state_dict(),
                's_state_dict': all_model.module.student.state_dict(),
                'best_acc1': best_acc1,
                'lbl': save_lbl,
                'pred': save_pred,
                }
        if  dist.get_rank() == 0 and is_best:
            torch.save(state, os.path.join(root, f'{check_path}/{config.env_name}_epoch_{epoch}_{acc}'))

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def train(train_loader, all_model,loss_scaler, optimizer, epoch, config,logger):
    losses = AverageMeter('Loss', ':.4e')
    loss_acc = AverageMeter('Acc', ':6.2f')
    
    
    progress = ProgressMeter(len(train_loader), [losses, loss_acc],
                              prefix="Epoch: [{}]".format(epoch), logger = logger)
    lamdakd = lambda_scheduler(config.kd_weight, epoch, alpha=config.kd_iter)
    lamdace = lambda_scheduler(config.ce_weight, epoch, alpha=config.ce_iter)
   
    
    all_model.module.to_train()
    
    prefetcher = data_prefetcher(train_loader)
    images, target, yy, bc, indices = prefetcher.next()
    i = 0
    optimizer.zero_grad()
    optimizer.step()
    logger.info(f"scale:{config.scale},tem:{config.T}")
    while images is not None:
        if epoch <1:
            out_s = all_model(images, config.T,is_train = False)
            loss = F.nll_loss(out_s['y'],target)
        else:
            if config.kd_mode==2:  # ours
                out_t, out_s = all_model(images, config.T,is_train = True)
                loss_kd = config.T * config.T *weight_funkd(torch.nn.KLDivLoss(),out_s['y_tem'], out_t['p'],F.nll_loss,out_t['y'],target,lamdakd,spl_type=config.spl_type)
                loss_ce=weight_funce(F.nll_loss,out_s['y'],target,lamdace,spl_type=config.spl_type)
                # loss_ce=weight_funce(torch.nn.CrossEntropyLoss(),out_s['logit'],target,lamdace,spl_type=config.spl_type)
                loss = loss_ce  + loss_kd *config.alpha 
            elif  config.kd_mode==1: # pace-weight
                out_s = all_model(images, config.T,is_train = False)
                #loss_kd = config.T * config.T *weight_funkd(torch.nn.KLDivLoss(),out_s['y_tem'], out_t['p'],F.nll_loss,out_t['y'],target,lamdakd,spl_type=config.spl_type)
                loss=weight_funce(F.nll_loss,out_s['y'],target,lamdace,spl_type=config.spl_type)   #+ loss_kd * config.alpha 
            else:# vallina kd
                out_t, out_s = all_model(images, config.T,is_train = True)
                criterion=torch.nn.KLDivLoss() 
                loss_kd = config.T * config.T *(criterion(out_s['y_tem'], out_t['p']))
                loss = F.nll_loss(out_s['y'], target) + loss_kd * config.alpha 
          

        acc = accuracy(out_s['y'],target)
        torch.distributed.barrier()
        reduced_loss = reduce_mean(loss, config.nprocs)
        reduced_acc = reduce_mean(acc, config.nprocs)
        
        losses.update(reduced_loss.item(), images.size(0))
        loss_acc.update(reduced_acc.item(), images.size(0))

        optimizer.zero_grad()
        loss_scaler(loss, optimizer, parameters=all_model.module.student.parameters())
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        optimizer.step()

        if i % config.print_freq == 0 and config.local_rank == 0:
            progress.display(i)

        i += 1

        images, target, yy, bc,indices = prefetcher.next()
    all_model.module.update_target()
    if config.local_rank == 0:
        logger.info(f"[train loss]: {round(float(losses.avg),4)}, [train acc]: {round(float(loss_acc.avg),4)}")


from sklearn.metrics import roc_auc_score,f1_score,confusion_matrix
def validate(val_loader, model, config, logger, prefix = ""):

    loss_metric = AverageMeter('acc', ':6.2f')
    progress = ProgressMeter(len(val_loader), [loss_metric], prefix='Test: ', logger = logger)
    model.module.to_eval()

    preds = []
    lbls = []
    outs = []
    lbl_onehot = np.zeros((len(lbls), len(np.unique(lbls))))
    for i in range(len(lbls)):
        lbl_onehot[i,lbls[i]]=1
    with torch.no_grad():
        prefetcher = data_prefetcher(val_loader)
        images, target, yy, bc,indices = prefetcher.next()
        while images is not None:
            out = model(images, config.T, is_train=False)
            acc = accuracy(out['y'], target)
            preds.extend(out['y'].max(1)[1].cpu().detach().numpy())
            lbls.extend(target.cpu().detach().numpy())
            outs.extend(out['y'].cpu().detach().numpy())
            torch.distributed.barrier()
            reduced_acc = reduce_mean(acc, config.nprocs)
            loss_metric.update(reduced_acc.item(), images.size(0))

            images, target, yy, bc, indices = prefetcher.next()
    auc = round(roc_auc_score(lbls,preds),4)
    cm = confusion_matrix(lbls,preds)
    eval_sen = round(cm[1, 1] / float(cm[1, 1]+cm[1, 0]),4)
    eval_spe = round(cm[0, 0] / float(cm[0, 0]+cm[0, 1]),4)
    if config.local_rank == 0:
        logger.info(f"\033[32m >>>>>>>> [{prefix}-acc]: {round(float(loss_metric.avg),4)}\
                    | auc: {auc} | recall: {eval_sen} | spe: {eval_spe} \033[0m")

    return round(loss_metric.avg,4),auc, np.array(lbls), np.array(outs)


if __name__ == '__main__':
    main()
