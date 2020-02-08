import argparse
from utils import print_args
from collections import defaultdict
import numpy as np

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Parse pytorch records')
    parser.add_argument('--fn', type=str,
                        help='the file name for the list of ')
    parser.add_argument('--axis', type=int,
                        help='the dimension for searching')
    args = parser.parse_args()
    # print_args(args)

    with open(args.fn) as fp:
        score = []
        for line in fp:
            line = line.strip().split()
            score.append(float(line[args.axis]))
        
    # print('mean, std = {:.3f}, {:.3f}, fn = {}'.format(np.mean(score), np.std(score), args.fn))
    print('& \\small {:.3f} $\\pm$ {:.3f}'.format(np.mean(score), np.std(score)))


def get_str(fn, axis):
    with open(fn) as fp:
        score = []
        for line in fp:
            line = line.strip().split()
            seed = line[0]
            score.append(float(line[axis]))
        
    # print('mean, std = {:.3f}, {:.3f}, fn = {}'.format(np.mean(score), np.std(score), args.fn))
    return '& \\small {:.3f} $\\pm$ {:.3f}'.format(np.mean(score), np.std(score))

def get_score(fn, axis):
    with open(fn) as fp:
        score = []
        for line in fp:
            line = line.strip().split()
            seed = line[0]
            score.append(float(line[axis]))
        
    return np.mean(score), np.std(score)

def get_hme_score(fn, axis):
    with open(fn) as fp:
        score = []
        for line in fp:
            line = line.strip().split()
            seed = line[0]
            score.append(float(line[axis]))
            break
        
    return np.mean(score), np.std(score)

if __name__ == '__main__':
    main()
