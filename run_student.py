from sys import prefix
import os
import torch
import logging
import random
import numpy as np
import warnings
import math
from args import get_parser
from dataset.data import data_prefetcher, AllData_DataFrame
import torch.distributed as dist
from models.sfcn_mini import SFCN
from models.dbnV2 import DBN
from models.resnetV2 import resnet18, resnet34, resnet50
from models.densenetV2 import densenet121, densenet201
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
root = "/data2/chenhr/CReg-KD-main"


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
def weight_func(func,x,y,lamda,spl_type):
    n=x.shape[0]
    ret = 0.
    for i in range(n):
        m=curr_v(func(x[i],y[i]),lamda,spl_type)
        ret += func(x[i],y[i]) *m
    return ret / n

def curr_w(l0,l, lamda):
        if l > lamda:
         v = 1./l0 -l / lamda
        else :
         v= 1.
        return v
def weight_ce(func,x,y,z,w,l):
    n=x.shape[0]
    ret = 0.
    cc=0.
    zz=0.
    for i in range(n):
        p=z[i] +1e-4
        entropy=-p[0]* math.log(p[0], 2)-p[1]* math.log(p[1], 2)
        ret += func(x[i],y[i])*curr_w(l,entropy,w)
        
    return ret / n
def main_worker(config, logger):
    model_names = ["resnet18", "resnet50", "dense121", "sfcn", "dbn"]
    models = [resnet18, resnet50, densenet121, SFCN, DBN]

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
    
    # all_model = DistributedDataParallel(all_model, find_unused_parameters=True)
    all_model = DistributedDataParallel(all_model, find_unused_parameters=True)
    all_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(all_model)
    
    optimizer = torch.optim.AdamW(all_model.module.student.parameters(),lr = config.lr,weight_decay = 0.0005)
    loss_scaler = NativeScaler()

    # all_model, optimizer = amp.initialize(all_model, optimizer, opt_level=config.opt_level)
    
    cudnn.benchmark = True

    # Data loading code
    train_data = AllData_DataFrame("/data5/yang/brain/dataset/pace_dataset.csv",config, train = True)
    val_data = AllData_DataFrame("/data5/yang/brain/dataset/pace_dataset.csv",config, train = False)
    test_data = AllData_DataFrame("/data5/yang/brain/dataset/pace_dataset.csv", config,train = False, test = True)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)

    train_loader = DataLoader(train_data,config.batch_size,
                        shuffle=False,num_workers=8,pin_memory = True, sampler = train_sampler)
    val_loader = DataLoader(val_data,config.batch_size,
                        shuffle=False,num_workers=8,pin_memory = False, sampler = val_sampler)
    test_loader = DataLoader(test_data,config.batch_size,
                       shuffle=False,num_workers=8,pin_memory = False, sampler = test_sampler)

    check_path = os.path.join(root,"checkpoints/mine/%s" % config.env_name)
    for epoch in range(config.epochs):
        # if epoch>0:
        #     use_file = os.path.join(root,check_path, f'{config.env_name}_last')
        #     checkpoint = torch.load(use_file, map_location='cpu') 
        #     all_model.module.load_previous_time(checkpoint['s_state_dict'])
        train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, config)
        import time
        # train for one epoch
        st_time = time.time()
        train(train_loader, all_model,loss_scaler, optimizer, epoch, config, logger)
        end_time = time.time()
        run_time = end_time - st_time
        logger.info(f"Running time for epoch: {run_time}")
        acc,auc = validate(val_loader,all_model, config, logger, prefix = "val")
        

        is_best = (acc > best_acc1) or (acc == best_acc1 and auc > best_auc)
        if is_best:
         test_acc,_ = validate(test_loader,all_model, config, logger, prefix = "test")
         print(test_acc)
         
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
                    # 'amp': amp.state_dict(),
                    # 'optimizer': optimizer.state_dict(),
                }
        if  dist.get_rank() == 0 and is_best:
          torch.save(state, os.path.join(root, f'{check_path}/{config.env_name}_epoch_{epoch}_{acc}'))
        # torch.save(state, os.path.join(root, f'{check_path}/{config.env_name}_last'))

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
    lamdakd = lambda_scheduler(config.spl_weight, epoch, alpha=config.kd_iter)
    lamdace = lambda_scheduler(config.entropy, epoch, alpha=config.ce_iter)
   
    
    all_model.module.to_train()
    
    prefetcher = data_prefetcher(train_loader)
    images, target, yy, bc, indices = prefetcher.next()
    i = 0
    optimizer.zero_grad()
    optimizer.step()
    logger.info(f"scale:{config.scale},tem:{config.T}")
    while images is not None:
        out_t, out_s = all_model(images, config.T)
        if epoch < 1:
          loss = F.nll_loss(out_s['y'], target)  
        else:
          if config.kd_mode==2:  # ours
             loss_kd = config.T * config.T *weight_func(torch.nn.KLDivLoss(),out_s['y_tem'], out_t['p'],lamdakd,spl_type=config.spl_type)
             loss_ce = weight_ce(F.nll_loss,out_s['y'],target,out_t['p'],lamdace,config.entropy)
             loss = loss_ce + loss_kd *config.alpha 
          elif  config.kd_mode==1: # pace-weight
             loss_kd = config.T * config.T *weight_func(torch.nn.KLDivLoss(),out_s['y_tem'], out_t['p'],lamdakd,spl_type=config.spl_type)
             loss=F.nll_loss(out_s['y'], target) + loss_kd * config.alpha 
          else:# vallina kd
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
        loss_scaler(loss, optimizer, parameters=all_model.module.student.parameters(),clip_grad=1,clip_mode='value')
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

    return round(loss_metric.avg,4),auc


if __name__ == '__main__':
    main()
