from __future__ import print_function

from data_utils import get_data_loaders, task_list, set_task
from utils import print_args, AverageMeter

import argparse


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Locally Constant Networks')
    # Task specification
    parser.add_argument('--dataset', type=str, required=True,
                        choices=task_list)
    parser.add_argument('--sub-task', type=int, default=1, 
                        help='sub-task')
     
    # parameters for tuning
    parser.add_argument('--depth', type=int, required=True,
                        help='the depth of the network (tree)')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--drop_type', type=str, default='none', 
                        choices=['node_dropconnect', 'none'])
    parser.add_argument('--p', type=float, default=0, 
                        help='the dropout rate, -1 means annealing from 1 - epoch / total epoch to 0.')

    # specific to the ensemble methods
    parser.add_argument('--ensemble_n', type=int, default=1,
                        help='the number of ensembles')
    parser.add_argument('--shrinkage', type=float, default=1.,
                        help='shrinkage of the boosting method')
    
    # parameters that do not require frequent tuning
    parser.add_argument('--back_n', type=int, default=0,
                        help='the depth of the backward network')
    parser.add_argument('--net_type', type=str, default='locally_constant', metavar='S',
                        choices=['locally_constant', 'locally_linear'])
    parser.add_argument('--hidden_dim', type=int, default=1,
                        help='the hidden dimension')
    parser.add_argument('--anneal', type=str, default='interpolation', choices=['interpolation', 'none', 'approx'],
                        help='whether to anneal ReLU')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        choices=['SGD', 'AMSGrad'])
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1), 0.1 seems good for molecule classification on fingerprints, may need to tune for other datasets / tasks.')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--lr_step_size', type=int, default=10,
                        help='How often to decrease learning by gamma.')
    parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')


    # the below are totally obsolete.... 
    parser.add_argument('--row_subsample', type=float, default=1.0, 
                        help='subsampling for gradient boosting')
    parser.add_argument('--col_subsample', type=float, default=1.0, 
                        help='subsampling for gradient boosting')
    
    args = parser.parse_args()
    args = set_task(args)
    # print_args(args)

    return args

