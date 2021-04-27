import os

import numpy as np
from recordclass import recordclass
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.optim as optim
from sklearn.tree import DecisionTreeRegressor
from torch.optim.lr_scheduler import StepLR

from data_utils import basic_loader
from network import Net
from train_utils import train, test


def get_alpha(epoch, total_epoch):
    return float(epoch) / float(total_epoch)


class LCNRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, depth=7, drop_type='node_dropconnect', dropout_rate=0.75, back_n=0, net_type='locally_constant',
                 hidden_dim=1, anneal='interpolation', optimizer='SGD', batch_size=64, epochs=30, lr=0.1, momentum=0.9,
                 no_cuda=False, lr_step_size=10, gamma=0.1, ensemble_n=1, shrinkage=1., seed=1):
        """
        depth: the depth of the network (tree)
        seed: random seed (default: 0)
        dropout_rate: the dropout rate, -1 means annealing from 1 - epoch / total epoch to 0.

        ensemble_n: the number of ensembles
        shrinkage: shrinkage of the boosting method

        back_n: the depth of the backward network
        net_type: ['locally_constant', 'locally_linear']
        hidden_dim: the hidden dimension
        anneal: whether to anneal ReLU
        optimizer: ['SGD', 'AMSGrad']
        batch_size: input batch size for training (default: 64)
        epochs: number of epochs to train (default: 10)
        lr: learning rate (default: 0.1), 0.1 seems good for molecule classification on fingerprints, may need to tune for other datasets / tasks.
        momentum: SGD momentum (default: 0.9)
        no_cuda: disables CUDA training
        lr_step_size: How often to decrease learning by gamma.
        gamma: LR is multiplied by gamma on schedule.
        """
        dictionary = {
            'dataset': 'dataset',
            # parameters for tuning
            'depth': depth,
            'seed': seed,
            'drop_type': drop_type,
            'p': dropout_rate,

            'ensemble_n': ensemble_n,
            'shrinkage': shrinkage,

            'back_n': back_n,
            'net_type': net_type,
            'hidden_dim': hidden_dim,
            'anneal': anneal,
            'optimizer': optimizer,
            'batch_size': batch_size,
            'epochs': epochs,
            'lr': lr,
            'momentum': momentum,
            'no_cuda': no_cuda,
            'lr_step_size': lr_step_size,
            'gamma': gamma,

            'input_dim': 1,
            'output_dim': 1,
            'task': 'regression',
        }

        self.args = recordclass("ObjectName", dictionary.keys())(*dictionary.values())

    def fit(self, X, y):
        x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

        # from some github repo...
        torch.multiprocessing.set_sharing_strategy('file_system')

        args = self.args

        args.input_dim = X.shape[1]
        args.output_dim = 1
        args.task = 'regression'

        use_cuda = not args.no_cuda and torch.cuda.is_available()

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        device = torch.device("cuda" if use_cuda else "cpu")
        self.device = device

        train_loader = basic_loader(x_train, y_train, args.batch_size)
        valid_loader = basic_loader(x_valid, y_valid, args.batch_size, train_shuffle=False)

        # train_loader, valid_loader, test_loader = get_data_loaders(args.dataset, args.batch_size,
        #                                                            sub_task=args.sub_task, dim=args.input_dim)

        # if args.dataset in ['sider_split/', 'tox21_split/']:
        #     args.dataset = args.dataset[:-1] + '-' + str(args.sub_task)

        print('batch number: train={}, valid={}'.format(len(train_loader), len(valid_loader)))

        model = Net(input_dim=args.input_dim, output_dim=args.output_dim, hidden_dim=args.hidden_dim,
                    num_layer=args.depth, num_back_layer=args.back_n, dense=True, drop_type=args.drop_type,
                    net_type=args.net_type, approx=args.anneal, device=device).to(device)
        self.model = model

        if args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)
        elif args.optimizer == 'AMSGrad':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
        scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

        best_score = -1e30
        start_epoch = 1  # start from epoch 1 or last checkpoint epoch
        if args.anneal == 'approx':
            args.net_type = 'approx_' + args.net_type

        best_model_name = './checkpoint/{}/{}/best_seed{}_depth{}_ckpt.t7'.format(args.dataset.strip('/'),
                                                                                  args.net_type, args.seed, args.depth)
        last_model_name = './checkpoint/{}/{}/last_seed{}_depth{}_ckpt.t7'.format(args.dataset.strip('/'),
                                                                                  args.net_type, args.seed, args.depth)

        best_log_file = 'log/' + args.dataset.strip('/') + '/{}/depth{}_backn{}_drop{}_p{}_best.log'.format(
            args.net_type, args.depth, args.back_n, args.drop_type, args.p)
        last_log_file = 'log/' + args.dataset.strip('/') + '/{}/depth{}_backn{}_drop{}_p{}_last.log'.format(
            args.net_type, args.depth, args.back_n, args.drop_type, args.p)

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
            # test_loss, test_score = test(args, model, device, test_loader, 'test')

            print(train_score, valid_score)
            # early stopping version
            if valid_score > best_score:
                self.best_state = model.state_dict()
                state = {'model': model.state_dict()}
                torch.save(state, best_model_name)
                best_score = valid_score

            # "convergent" version
            state = {'model': model.state_dict()}
            torch.save(state, last_model_name)

        # print('Training finished. Loading models from validation...')
        # for model_name, log_file, setting in zip([best_model_name, last_model_name], [best_log_file, last_log_file],
        #                                          ['best', 'last']):
        #     print('\nLoading the {} model...'.format(setting))
        #
        #     checkpoint = torch.load(model_name)
        #     model.load_state_dict(checkpoint['model'])
        #     train_loss, train_score = test(args, model, device, train_loader, 'train')
        #     valid_loss, valid_score = test(args, model, device, valid_loader, 'valid')
        # test_loss, test_score = test(args, model, device, test_loader, 'test ')
        return self

    def predict(self, X):
        args = self.args
        test_x = X
        test_y = np.zeros(X.shape[0])
        test_loader = basic_loader(test_x, test_y, args.batch_size, train_shuffle=False)
        model = self.model
        model.load_state_dict(self.best_state)
        test_loss, test_score, output = test(args, self.model, self.device, test_loader, 'test', return_output=True)
        return torch.cat(output, dim=0).numpy()


if __name__ == '__main__':
    X, y = load_boston(return_X_y=True)
    x_scaler = StandardScaler()
    X = x_scaler.fit_transform(X)
    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y.reshape(-1, 1))
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    l = LCNRegressor(depth=3, epochs=50, optimizer='AMSGrad', no_cuda=True)
    l.fit(x_train, y_train)
    print(r2_score(y_scaler.inverse_transform(y_train), y_scaler.inverse_transform(l.predict(x_train))))
    print(r2_score(y_scaler.inverse_transform(y_test), y_scaler.inverse_transform(l.predict(x_test))))

    l = DecisionTreeRegressor(max_depth=3)
    l.fit(x_train, y_train)
    print(r2_score(y_scaler.inverse_transform(y_train), y_scaler.inverse_transform(l.predict(x_train))))
    print(r2_score(y_scaler.inverse_transform(y_test), y_scaler.inverse_transform(l.predict(x_test))))
