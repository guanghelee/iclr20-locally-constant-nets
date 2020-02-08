from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import AverageMeter
import numpy as np
from sklearn.metrics import roc_auc_score

def train(args, model, device, train_loader, optimizer, epoch, anneal, alpha=1):
    model.train()
    dataset_len = 0
    avg_loss = AverageMeter()

    for (data, target) in train_loader:
        dataset_len += len(target)
        data, target = data.to(device), target.to(device)
        if args.task == 'classification':
            target = target.type(torch.cuda.LongTensor)
                
        optimizer.zero_grad()
        ###############
        data.requires_grad = True
        if model.net_type == 'locally_constant':
            if args.p != -1:
                assert(args.p >= 0. and args.p < 1)
                output, regularization = model(data, alpha=alpha, anneal=anneal, p=args.p, training=True)
            else:
                output, regularization = model(data, alpha=alpha, anneal=anneal, p=1-alpha, training=True)

        elif model.net_type == 'locally_linear':
            output, regularization = model.normal_forward(data)
        ###############
        
        optimizer.zero_grad()
        if args.task == 'classification':
            loss = F.cross_entropy(output, target)
        elif args.task == 'regression':
            output = output.squeeze(-1)
            loss = ((output - target) ** 2).mean() 
            
        loss.backward()
        optimizer.step()
        avg_loss.update(loss.item())
        
    return avg_loss.avg

def test(args, model, device, test_loader, test_set_name):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        
        score = []
        label = []
        dataset_len = 0

        pattern_to_pred = dict()
        tree_x = []
        tree_pattern = []

        for data, target in test_loader:
            dataset_len += len(target)
            label += list(target)
            data, target = data.to(device), target.to(device)
            if args.task == 'classification':
                target = target.type(torch.cuda.LongTensor)
            
            ###############
            data.requires_grad = True
            if model.net_type == 'locally_constant':
                output, relu_masks = model(data, p=0, training=False)
            elif model.net_type == 'locally_linear':
                output, relu_masks = model.normal_forward(data, p=0, training=False)
            ###############

            if args.task == 'classification':
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                output = torch.softmax(output, dim=-1)
                score += list(output[:, 1].cpu().data.numpy())
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                output = output[:, 1]
            elif args.task == 'regression':
                output = output.squeeze(-1)
                test_loss += ((output - target) ** 2).mean().item() * len(target)

        test_loss /= dataset_len
        if args.task == 'classification':
            if args.output_dim == 2:
                AUC = roc_auc_score(label, score)
                test_score = AUC
            else:
                AUC = -1
                test_score = correct / dataset_len
            
        elif args.task == 'regression':
            RMSE = np.sqrt(test_loss)
            test_score = -RMSE
            
        return test_loss, test_score
        