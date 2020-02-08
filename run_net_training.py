from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from network import my_softplus, my_softplus_derivative, Net
from train_utils import train, test
from data_utils import get_data_loaders, task_list, set_task, save_pklgz, get_train_np_loader
from utils import print_args, AverageMeter
from arg_utils import get_args

import numpy as np
import os
from time import time
import argparse
from sklearn.metrics import roc_auc_score

def get_alpha(epoch, total_epoch):
    return float(epoch) / float(total_epoch)

def main():
    # from some github repo...
    torch.multiprocessing.set_sharing_strategy('file_system')

    args = get_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available() 

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, valid_loader, test_loader = get_data_loaders(args.dataset, args.batch_size, sub_task=args.sub_task, dim=args.input_dim)

    if args.dataset in ['sider_split/', 'tox21_split/']:
        args.dataset = args.dataset[:-1] + '-' + str(args.sub_task)

    print('batch number: train={}, valid={}, test={}'.format(len(train_loader), len(valid_loader), len(test_loader)))

    model = Net(input_dim=args.input_dim, output_dim=args.output_dim, hidden_dim=args.hidden_dim, num_layer=args.depth, num_back_layer=args.back_n, dense=True, drop_type=args.drop_type, net_type=args.net_type, approx=args.anneal).to(device)
    
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)
    elif args.optimizer == 'AMSGrad':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)
    
    best_score = -1e30
    start_epoch = 1  # start from epoch 1 or last checkpoint epoch
    if args.anneal == 'approx':
        args.net_type = 'approx_' + args.net_type 


    best_model_name = './checkpoint/{}/{}/best_seed{}_depth{}_ckpt.t7'.format(args.dataset.strip('/'), args.net_type, args.seed, args.depth)
    last_model_name = './checkpoint/{}/{}/last_seed{}_depth{}_ckpt.t7'.format(args.dataset.strip('/'), args.net_type, args.seed, args.depth)
    
    best_log_file = 'log/' + args.dataset.strip('/') + '/{}/depth{}_backn{}_drop{}_p{}_best.log'.format(args.net_type, args.depth, args.back_n, args.drop_type, args.p)
    last_log_file = 'log/' + args.dataset.strip('/') + '/{}/depth{}_backn{}_drop{}_p{}_last.log'.format(args.net_type, args.depth, args.back_n, args.drop_type, args.p)

    model_dir = './checkpoint/{}/{}/'.format(args.dataset.strip('/'), args.net_type)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)  
    log_dir = 'log/' + args.dataset.strip('/') + '/{}/'.format(args.net_type)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  
    
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')

    for epoch in range(start_epoch, args.epochs + start_epoch):
        scheduler.step(epoch)

        alpha = get_alpha(epoch, args.epochs)
        train_approximate_loss = train(args, model, device, train_loader, optimizer, epoch, args.anneal, alpha)

        # used for plotting learning curves
        train_loss, train_score = test(args, model, device, train_loader, 'train')
        valid_loss, valid_score = test(args, model, device, valid_loader, 'valid')
        test_loss, test_score = test(args, model, device, test_loader, 'test')
        
        # early stopping version
        if valid_score > best_score:
            state = {'model': model.state_dict()}
            torch.save(state, best_model_name)
            best_score = valid_score

        # "convergent" version
        state = {'model': model.state_dict()}
        torch.save(state, last_model_name)
        
    print('Training finished. Loading models from validation...')
    for model_name, log_file, setting in zip([best_model_name, last_model_name], [best_log_file, last_log_file], ['best', 'last']):
        print('\nLoading the {} model...'.format(setting))

        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint['model'])
        train_loss, train_score = test(args, model, device, train_loader, 'train')
        valid_loss, valid_score = test(args, model, device, valid_loader, 'valid')
        test_loss, test_score = test(args, model, device, test_loader, 'test ')
        
        with open(log_file, 'a') as fp:
            if args.task == 'classification':
                log_str = '{}\t{:.4f}\t{:.4f}\t{:.4f}'.format(args.seed, train_score, valid_score, test_score)
            elif args.task == 'regression':
                log_str = '{}\t{:.4f}\t{:.4f}\t{:.4f}'.format(args.seed, -train_score, -valid_score, -test_score)
            fp.write(log_str+'\n')


if __name__ == '__main__':
    main()

