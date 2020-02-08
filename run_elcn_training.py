from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from network import my_softplus, my_softplus_derivative, Net
from data_utils import get_data_loaders, task_list, set_task, save_pklgz, get_train_np_loader
from utils import print_args, AverageMeter
from arg_utils import get_args

import numpy as np
import os
from time import time
import argparse
from sklearn.metrics import roc_auc_score

def get_labels(loaders):
    label_list = []
    for loader in loaders:
        labels = []
        for data, target in loader:
            labels += list(target)
        label_list.append(np.array(labels))
    return label_list

def get_AUC(train_pred, train_labels):
    pred = np.sum(train_pred, axis=1)
    pred = F.softmax(torch.tensor(pred), dim=-1)[:, 1]
    pred = pred.data.numpy()
    # print(pred.shape, train_labels.shape)
    # [:, 1]
    return roc_auc_score(train_labels, pred)

def get_RMSE(train_pred, train_labels, offset):
    pred = (np.sum(train_pred, axis=1).flatten() + offset[0])
    return np.sqrt(np.mean((pred - train_labels.flatten()) ** 2))

def train(args, model, device, train_loader, optimizer, epoch, anneal, train_pred, offset, feat_indices, data_indices, alpha=1):
    model.train()
    dataset_len = 0
    avg_loss = AverageMeter()

    train_x = train_loader.dataset.x[:, feat_indices]
    train_y = train_loader.dataset.y

    np.random.shuffle(data_indices)

    for data_idx in range(0, len(data_indices), args.batch_size):
        data = torch.tensor(train_x[data_indices[data_idx:data_idx+args.batch_size]]).to(device)
        target = torch.tensor(train_y[data_indices[data_idx:data_idx+args.batch_size]]).to(device)

        base = np.sum(train_pred[data_indices[data_idx:data_idx+args.batch_size]], axis=1) + offset
        base = torch.tensor(base).type(torch.cuda.FloatTensor)

        if args.task == 'classification':
            target = target.type(torch.cuda.LongTensor)
                
        optimizer.zero_grad()
        ###############
        data.requires_grad = True
        if args.p != -1:
            assert(args.p >= 0. and args.p < 1)
            output, regularization = model(data, alpha=alpha, anneal=anneal, p=args.p, training=True)
        else:
            output, regularization = model(data, alpha=alpha, anneal=anneal, p=1-alpha, training=True)
        ###############

        output = output + base
        
        optimizer.zero_grad()
        if args.task == 'classification':
            # loss = F.nll_loss(torch.log(output), target) 
            loss = F.cross_entropy(output, target)
            # + 0. * regularization
        elif args.task == 'regression':
            output = output.squeeze(-1)
            loss = ((output - target) ** 2).mean() 
            # + 0. * regularization

        loss.backward()
        optimizer.step()
        avg_loss.update(loss.item())
        
    return avg_loss.avg

def predict(args, model, device, test_loader, feat_indices):
    model.eval()
    score = []
    for data, target in test_loader:
        data = data[:, feat_indices]
        data, target = data.to(device), target.to(device)
        if args.task == 'classification':
            target = target.type(torch.cuda.LongTensor)

        ###############
        data.requires_grad = True
        if model.net_type == 'locally_constant':
            output, relu_mask = model(data)
        elif model.net_type == 'locally_linear':
            output, relu_mask = model.normal_forward(data)
        ###############

        if args.task == 'classification':
            score += list(output.cpu().data.numpy())
        elif args.task == 'regression':
            score += list(output.cpu().data.numpy())

    return np.array(score)

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

    train_loader, valid_loader, test_loader = get_data_loaders(args.dataset, args.batch_size, sub_task=args.sub_task, dim=args.input_dim, train_shuffle=False)
    train_labels, valid_labels, test_labels = get_labels([train_loader, valid_loader, test_loader])

    if args.dataset in ['sider_split/', 'tox21_split/']:
        args.dataset = args.dataset[:-1] + '-' + str(args.sub_task)

    print('batch number: train={}, valid={}, test={}'.format(len(train_loader), len(valid_loader), len(test_loader))) 
    
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')

    train_pred = np.zeros((len(train_labels), args.ensemble_n, args.output_dim), dtype=np.float32)
    valid_pred = np.zeros((len(valid_labels), args.ensemble_n, args.output_dim), dtype=np.float32)
    test_pred  = np.zeros((len(test_labels), args.ensemble_n, args.output_dim), dtype=np.float32)

    if args.task == 'classification':
        offset = np.array([[0., 0.]], dtype=np.float32)
    else:
        offset = np.array([[0.]], dtype=np.float32)

    ckpt_dir = 'checkpoint/' + args.dataset.strip('/') + '/ensemble/'
    log_dir = 'log/' + args.dataset.strip('/') + '/ensemble/'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)  
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  

    ckpt_file = ckpt_dir + 'depth{}_backn{}_drop{}_p{}_shrinkage{}_seed{}.ckpt'.format(args.depth, args.back_n, args.drop_type, args.p, args.shrinkage, args.seed)
    best_file = ckpt_dir + 'depth{}_backn{}_drop{}_p{}_shrinkage{}_seed{}.t7'.format(args.depth, args.back_n, args.drop_type, args.p, args.shrinkage, args.seed)
    log_file  = log_dir  + 'depth{}_backn{}_drop{}_p{}_shrinkage{}_seed{}.log'.format(args.depth, args.back_n, args.drop_type, args.p, args.shrinkage, args.seed) 

    for ensemble_idx in range(args.ensemble_n):
        feat_indices = np.arange(args.input_dim)
        feat_dim = args.input_dim
        data_indices = np.arange(len(train_loader.dataset.x))
        model = Net(input_dim=feat_dim, output_dim=args.output_dim, hidden_dim=args.hidden_dim, num_layer=args.depth, num_back_layer=args.back_n, dense=True, drop_type=args.drop_type, net_type=args.net_type, approx=args.anneal).to(device)
    
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)
        elif args.optimizer == 'AMSGrad':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
        scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)
        
        best_score = -1e30
        start_epoch = 1  # start from epoch 1 or last checkpoint epoch

        start = time()
        if ensemble_idx == 0 and args.task == 'regression':
            shrinkage = 1
        else:
            shrinkage = args.shrinkage

        for epoch in range(start_epoch, args.epochs + start_epoch):
            scheduler.step(epoch)

            alpha = get_alpha(epoch, args.epochs)
            train_approximate_loss = train(args, model, device, train_loader, optimizer, epoch, args.anneal, train_pred, offset, feat_indices, data_indices, alpha)
            if epoch % 30 == 0:
                print('Train Epoch: {} \tLoss: {:.6f}'.format(
                    epoch, train_approximate_loss), flush=True)

        train_pred[:, ensemble_idx, :] = shrinkage * predict(args, model, device, train_loader, feat_indices)
        valid_pred[:, ensemble_idx, :] = shrinkage * predict(args, model, device, valid_loader, feat_indices)
        test_pred[:, ensemble_idx, :]  = shrinkage * predict(args, model, device, test_loader, feat_indices)

        save_pklgz(ckpt_file, [train_pred, train_labels, valid_pred, valid_labels, test_pred, test_labels])
        if args.task == 'classification':
            train_score = get_AUC(train_pred, train_labels)
            valid_score = get_AUC(valid_pred, valid_labels)
            test_score  = get_AUC(test_pred,  test_labels)
            print('Iteration {}, AUC: train = {:.3f}, valid = {:.3f}, test = {:.3f}'.format(ensemble_idx, train_score, valid_score, test_score))
        else:
            train_score = get_RMSE(train_pred, train_labels, offset)
            valid_score = get_RMSE(valid_pred, valid_labels, offset)
            test_score  = get_RMSE(test_pred, test_labels, offset)
            print('Iteration {}, RMSE: train = {:.3f}, valid = {:.3f}, test = {:.3f}'.format(ensemble_idx, train_score, valid_score, test_score))
        
        with open(log_file, 'a') as fp:
            fp.write('{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(args.seed, ensemble_idx, train_score, valid_score, test_score))
        del model, optimizer, scheduler

if __name__ == '__main__':
    main()

