import argparse

def get_parser():
    model_names = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data', metavar='DIR', default='/home/amax/data/yangyanwu/sub_mm', help='path to dataset')
    parser.add_argument('-a',
                        '--arch',
                        metavar='ARCH',
                        default='resnet18',
                        choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
    parser.add_argument('--epochs', default=180, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--fold', default=0, type=int, help='fold number')
    parser.add_argument('--da',action='store_true', help='fold number')
    parser.add_argument('--pad',action='store_true', help='fold number')
    parser.add_argument('-b',
                        '--batch-size',
                        default=32,
                        type=int,
                        metavar='N',
                        help='mini-batch size (default: 6400), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr',
                        '--learning-rate',
                        default=1e-6,
                        type=float,
                        metavar='LR',
                        help='initial learning rate',
                        dest='lr')
    parser.add_argument('--local-rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--wd',
                        '--weight-decay',
                        default=0.00005,
                        type=float,
                        metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=50, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
    parser.add_argument('--lam', default=0.5, type=float, help='lam value. ')
    
    parser.add_argument('--T', default=1, type=float, help='alpha value. ')
    parser.add_argument('--ab', default=0, type=int, help='alpha value. ')
    parser.add_argument('--beta', default=0.8, type=float, help='alpha value. ')
    parser.add_argument('--alpha', default=1, type=float, help='alpha value. ')
    parser.add_argument('--scale', default=5, type=float, help='alpha value. ')
   
    parser.add_argument('--ce_weight', default=0.6, type=float, help='spl kd weight. ')
    parser.add_argument('--kd_weight', default=0.8, type=float, help='spl kd weight. ')
    parser.add_argument('--kd_mode', default=2, type=int, help='training mode. 0: KD, 1: pace_weigth, 2: ours ')
    parser.add_argument('--spl_type', default=1, type=int, help='training mode. 0: hard, 1: soft, 2: log ')
    parser.add_argument('--ce_iter', default=0.006, type=int, help='ce iter weight. ')
    parser.add_argument('--kd_iter', default=0.003, type=int, help='kd iter weight. ')
    
    
    parser.add_argument('--mode', default=0, type=int, help='training mode.' )
    parser.add_argument('--env_name', default = "default", help='name for env')
    parser.add_argument('--opt_level', default = "O1", help='opt level, O1 default')
    parser.add_argument('--samples', '-s', default = 200, type=int, help='opt level, O1 default')

    args = parser.parse_args()
    args.env_name = f"{args.env_name}_fold-{args.fold}-model-{args.arch}-samples-{args.samples}"
    
    #args.env_name +  "-model.%s-mode.%s-lam.%s-alpha.%s-beta.%s-scale.%s-T.%s" % \
    #    (args.arch, args.mode, args.lam, args.alpha, args.beta, args.scale, args.T)
    return args
