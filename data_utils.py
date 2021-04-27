import numpy as np
import gzip, pickle
import torch
from torch.utils.data import Dataset, DataLoader

task_list = ['bace_split/', 'HIV_split/', 'sider_split/', 'tox21_split/', 'PDBbind']


def set_task(args):
    if args.dataset in ['bace_split/', 'HIV_split/', 'sider_split/', 'tox21_split/']:
        args.input_dim = 2048
        args.output_dim = 2
        args.task = 'classification'
    elif args.dataset == 'PDBbind':
        args.input_dim = 2052
        args.output_dim = 1
        args.task = 'regression'
    return args


def save_pklgz(filename, obj):
    pkl = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL);
    with gzip.open(filename, "wb") as fp:
        fp.write(pkl);


def load_pklgz(filename):
    print("... loading", filename);
    with gzip.open(filename, 'rb') as fp:
        obj = pickle.load(fp);
    return obj;


class Feat(Dataset):
    def __init__(self, x, y):
        assert (x.shape[0] == y.shape[0])
        self.x = x.astype(np.float32)
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def load_chem_data(fn, sub_task):
    x = []
    y = []
    with open(fn) as fp:
        fp.readline()
        for line in fp:
            line = line.strip().split(',')

            fgp = []
            for idx in range(len(line[0])):
                if line[0][idx] == '0':
                    fgp.append(0.)
                else:
                    fgp.append(1.)

            if line[sub_task] == '':
                continue
            y.append(int(line[sub_task]))
            x.append(np.array(fgp))

    return np.array(x).astype(np.float64), np.array(y)


def get_train_np_loader(dir_name, batch_size, sub_task=1, num_workers=1, pin_memory=False, dim=2048, fit=False,
                        return_np=False):
    if dir_name in ['bace_split/', 'HIV_split/', 'sider_split/', 'tox21_split/']:
        dir_name = 'data/' + dir_name
        train_x, train_y = load_chem_data(dir_name + 'train.fgp{}.csv'.format(dim), sub_task)
    elif dir_name == 'PDBbind':
        train_x, train_y, valid_x, valid_y, test_x, test_y = load_pdbbind_grid()
    return train_x, train_y


def get_data_loaders(dir_name, batch_size, sub_task=1, num_workers=0, pin_memory=False, dim=2048, fit=False,
                     return_np=False, train_shuffle=True):
    if dir_name in ['bace_split/', 'HIV_split/', 'sider_split/', 'tox21_split/']:
        dir_name = 'data/' + dir_name
        train_x, train_y = load_chem_data(dir_name + 'train.fgp{}.csv'.format(dim), sub_task)
        valid_x, valid_y = load_chem_data(dir_name + 'valid.fgp{}.csv'.format(dim), sub_task)
        test_x, test_y = load_chem_data(dir_name + 'test.fgp{}.csv'.format(dim), sub_task)
    elif dir_name == 'PDBbind':
        train_x, train_y, valid_x, valid_y, test_x, test_y = load_pklgz('data/PDBbind.pkl.gz')

    print('train, valid, and test data shapes (x.shape, y.shape)')
    print(train_x.shape, train_y.shape)
    print(valid_x.shape, valid_y.shape)
    print(test_x.shape, test_y.shape)

    if len(set(test_y)) < 50:
        for y in set(test_y):
            print('pred = {}, ACC = {:.4f}'.format(y, len(np.where(test_y == y)[0]) / len(test_y)))
    else:
        print('train RMSE = {:.4f}'.format(np.sqrt(np.mean((train_y - np.mean(train_y)) ** 2))))
        print('test  RMSE = {:.4f}'.format(np.sqrt(np.mean((test_y - np.mean(train_y)) ** 2))))

    if return_np:
        return train_x, train_y, valid_x, valid_y, test_x, test_y

    return data_to_loader(train_x, train_y, valid_x, valid_y, test_x, test_y,
                          batch_size, num_workers, pin_memory, train_shuffle)


def data_to_loader(train_x, train_y, valid_x, valid_y, test_x, test_y,
                   batch_size, num_workers=0, pin_memory=False, train_shuffle=True):
    train_loader = basic_loader(train_x, train_y, batch_size, num_workers, pin_memory, train_shuffle)
    valid_loader = basic_loader(valid_x, valid_y, batch_size, num_workers, pin_memory, train_shuffle=False)
    test_loader = basic_loader(test_x, test_y, batch_size, num_workers, pin_memory, train_shuffle=False)
    return train_loader, valid_loader, test_loader


def basic_loader(train_x, train_y, batch_size, num_workers=0, pin_memory=False, train_shuffle=True):
    train_data = Feat(train_x, train_y)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=train_shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader


def main():
    train_x, train_y, valid_x, valid_y, test_x, test_y = get_data_loaders('PDBbind', 32, return_np=True)


if __name__ == '__main__':
    main()
