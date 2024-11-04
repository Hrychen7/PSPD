from sys import prefix
import os
import torch
import logging
import math
import random
import numpy as np
import warnings
from args import get_parser
from dataset.data import data_prefetcher, AllData_DataFrame
import torch.distributed as dist
from models.sfcn_mini import SFCN
from models.dbnV2 import DBN
from models.resnetV2 import resnet18, resnet34, resnet50, resnet101
from models.densenetV2 import densenet121, densenet201
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import models.context
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
def lambda_scheduler(lambda_0, iter, alpha=0.0001, iter_0=100):
    if iter < iter_0:
        lamda = lambda_0 + alpha * iter
    else:
        lamda = lambda_0 + alpha * iter_0
    return lamda
def curr_v(l, lamda, spl_type=0):
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
    return g
def weight_func(func,x,y,lamda,spl_type):
    n=x.shape[0]
    ret = 0.
    for i in range(n):
        ret += func(x[i],y[i]) *curr_v(func(x[i],y[i]),lamda,spl_type)
    return ret / n
def get_loss(fun,x,y,epoch,ratio):
    n=x.shape[0]
    ret=[]
    loss=0.
    for i in range(n):
        ret.append(fun(x[i][None],y[i][None]))
    ret = sorted(ret)
    nn=min(n, n*ratio+epoch//5)
    for i in range(int(nn)):
        loss +=ret[i]
    return loss/n
def main_worker(config, logger):
    model_names = ["resnet18", "resnet50", "resnet101", "sfcn", "dbn"]
    models = [resnet18, resnet18, resnet101, SFCN, DBN]

    best_acc1 = -99.0
    best_acc2 = -99.0
    best_auc = -99.0
    dist.init_process_group(backend='nccl')
    # create model
    model_t = models[model_names.index(config.arch)](output_dim=2, mode = config.mode)
    model_s = models[model_names.index(config.arch)](output_dim=2, mode = config.mode)
    
    
    torch.cuda.set_device(config.local_rank)

    # find the best teacher epoch
    # dirs = f"checkpoints/default_fold-{config.fold}-model-{config.arch}"
    if config.arch == 'resnet18':
        dirs = f"checkpoints/baseline_resnet18_fold{config.fold}_fold-{config.fold}-model-resnet18-samples-200"
    elif config.arch == 'resnet50':
        dirs = f"checkpoints/baseline_resnet50_fold{config.fold}_fold-{config.fold}-model-resnet18-samples-200"
    elif config.arch == 'resnet101':
        dirs = f"checkpoints/baseline_resnet101_fold{config.fold}_fold-{config.fold}-model-resnet101-samples-200"
    
    files = os.listdir(os.path.join(root,dirs))
    trained_epoch = [int(f.split("_")[-2]) for f in files]
    max_epoch = max(trained_epoch)
    use_file = files[trained_epoch.index(max_epoch)]
    print(use_file)

    checkpoint = torch.load(os.path.join(root,dirs,use_file), map_location='cpu') 
    model_t.load_state_dict(checkpoint['state_dict'])

    # train_list = torch.nn.ModuleList()
    # train_list.append(model_s)
    
    
    
    model_s.cuda()
    model_t.cuda()
    

    logger.info("Loading finished.")
    config.batch_size = int(config.batch_size / config.nprocs)
    
    # optimizer = torch.optim.Adam(train_list.parameters(),lr = config.lr,weight_decay = 0.0005)

    # train_list, optimizer = amp.initialize(train_list, optimizer, opt_level=config.opt_level)
    # model_s = DistributedDataParallel(model_s)
    optimizer = torch.optim.Adam(model_s.parameters(),lr = config.lr,weight_decay = 0.0001)
    model_s = DistributedDataParallel(model_s,find_unused_parameters=True)
    
    cudnn.benchmark = True

    # Data loading code
    train_data = AllData_DataFrame("dataset/pace_dataset.csv",config, train = True)
    val_data = AllData_DataFrame("dataset/pace_dataset.csv",config, train = False)
    test_data = AllData_DataFrame("dataset/pace_dataset.csv", config,train = False, test = True)


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
    best_test_acc = -1
    save_lbl = 0
    save_pred = 0
    for epoch in range(config.epochs):
        train_sampler.set_epoch(epoch)

        # adjust_learning_rate(optimizer, epoch, config)
        import time
        # train for one epoch
        st_time = time.time()
        # dirs = f"checkpoints/default_fold-{config.fold}-model-{config.arch}"
        # dirs = f"baseline_resnet18_fold{config.fold}_fold-{config.fold}-model-resnet18-samples-200"
        # files = os.listdir(os.path.join(root,dirs))
        # trained_epoch = [int(f.split("_")[-2]) for f in files]
        # use_file = files[trained_epoch.index(epoch-1)]
        # print(use_file)

        # checkpoint = torch.load(os.path.join(root,dirs,use_file), map_location='cpu') 
        # model_t.load_state_dict(checkpoint['state_dict'])
        # model_t.cuda().eval()
        train(train_loader, model_t, model_s, optimizer, epoch, config, logger)
        end_time = time.time()
        run_time = end_time - st_time
        logger.info(f"Running time for epoch: {run_time}")
        acc,auc,_,_ = validate(val_loader,model_s, config, logger, prefix = "val")
       
        is_best = (acc > best_acc1) or (acc == best_acc1 and auc > best_auc)
        if is_best:
            # test_acc,_ = validate(test_loader,model_t, config, logger, prefix = "test")
            test_acc,_, test_lbl, test_pred = validate(test_loader,model_s, config, logger, prefix = "test")
            best_test_acc = test_acc
            save_lbl = test_lbl
            save_pred = test_pred
        # print("best acc: ", best_test_acc)
        logger.info(f"best acc: {best_test_acc:.4f}")
        best_acc1 = max(acc, best_acc1)
        best_auc = max(auc, best_auc)
        
        
        if not os.path.exists(check_path):
            try:
                os.makedirs(check_path)
            except:
                pass # multiple processors bug

        if  config.local_rank == 0 :
            state = {
                    'epoch': epoch + 1,
                    'state_dict': model_s.module.state_dict(),
                    'best_acc1': best_acc1,
                    'lbl': save_lbl,
                    'pred': save_pred,
                }
            torch.save(state, os.path.join(root, f'{check_path}/{config.env_name}_epoch_{epoch}_{acc}'))

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def train(train_loader, model_t,model_s, optimizer, epoch, config,logger):
    losses = AverageMeter('Loss', ':.4e')
    loss_acc = AverageMeter('Acc', ':6.2f')

    progress = ProgressMeter(len(train_loader), [losses, loss_acc],
                             prefix="Epoch: [{}]".format(epoch), logger = logger)
    lamda = lambda_scheduler(config.kd_weight, epoch, alpha=config.kd_iter)
    model_t.eval()
    model_s.train()

    prefetcher = data_prefetcher(train_loader)
    images, target, yy, bc, indices = prefetcher.next()
    i = 0
    optimizer.zero_grad()
    optimizer.step()
    logger.info(f"scale:{config.scale},tem:{config.T}")
    while images is not None:
        out_s = model_s(images, config.T)
        out_t = model_t(images, config.T)
        
        # weight = abs(out_t['p'].max(1)[0] - target) / config.scale
        # weight[weight >1] = 1
        # weight = 1 - weight
        
        loss_kd = config.T * config.T *weight_func(torch.nn.KLDivLoss(),out_s['y_tem'], out_t['p'],lamda,spl_type=1)
        loss = F.nll_loss(out_s['y'], target) + loss_kd * config.alpha 

        acc = accuracy(out_s['y'],target)
        torch.distributed.barrier()
        reduced_loss = reduce_mean(loss, config.nprocs)
        reduced_acc = reduce_mean(acc, config.nprocs)
        
        losses.update(reduced_loss.item(), images.size(0))
        loss_acc.update(reduced_acc.item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        optimizer.step()

        if i % config.print_freq == 0 and config.local_rank == 0:
            progress.display(i)

        i += 1

        images, target, yy, bc,indices = prefetcher.next()

    if config.local_rank == 0:
        logger.info(f"[train loss]: {round(float(losses.avg),4)}, [train acc]: {round(float(loss_acc.avg),4)}")

from sklearn.metrics import roc_auc_score,f1_score,confusion_matrix
def validate(val_loader, model, config, logger, prefix = ""):

    loss_metric = AverageMeter('acc', ':6.2f')
    progress = ProgressMeter(len(val_loader), [loss_metric], prefix='Test: ', logger = logger)
    model.eval()

    preds = []
    outs = []
    lbls = []
    lbl_onehot = np.zeros((len(lbls), len(np.unique(lbls))))
    for i in range(len(lbls)):
        lbl_onehot[i,lbls[i]]=1
    with torch.no_grad():
        prefetcher = data_prefetcher(val_loader)
        images, target, yy, bc,indices = prefetcher.next()
        while images is not None:
            out = model(images)['y']
            acc = accuracy(out, target)
            #loss_auc = roc_auc_score()
            outs.extend(out.cpu().detach().numpy())
            preds.extend(out.max(1)[1].cpu().detach().numpy())
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

    return round(loss_metric.avg,4),auc,np.array(lbls),np.array(outs)


if __name__ == '__main__':
    main()
