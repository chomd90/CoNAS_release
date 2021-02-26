"""
CoNAS search algorithm.
"""

import sys
import os
import shutil
import logging
import pickle
import argparse
import numpy as np
import utils
import torchvision.datasets as dset
import torch.nn as nn
import torch
import psr

from genotypes import mask_to_str

device = torch.device("cuda")

class Rung:
    def __init__(self, rung, nodes):
        self.parents = set()
        self.children = set()
        self.rung = rung
        for node in nodes:
            n = nodes[node]
            if n.rung == self.rung:
                self.parents.add(n.parent)
                self.children.add(n.node_id)

class Node:
    def __init__(self, parent, arch, node_id, rung):
        self.parent = parent
        self.arch = arch
        self.node_id = node_id
        self.rung = rung
    def to_dict(self):
        out = {'parent':self.parent, 'arch': self.arch, 'node_id': self.node_id, 'rung': self.rung}
        if hasattr(self, 'objective_val'):
            out['objective_val'] = self.objective_val
        return out

class Random_NAS:
    def __init__(self, B, model, seed, save_dir):
        self.save_dir = save_dir

        self.B = B
        self.model = model
        self.seed = seed

        self.iters = 0

        self.arms = {}
        self.node_id = 0

    def print_summary(self):
        logging.info(self.parents)
        objective_vals = [(n,self.arms[n].objective_val) for n in self.arms if hasattr(self.arms[n],'objective_val')]
        objective_vals = sorted(objective_vals,key=lambda x:x[1])
        best_arm = self.arms[objective_vals[0][0]]
        val_ppl = self.model.evaluate(best_arm.arch, split='valid')
        logging.info(objective_vals)
        logging.info('best valid ppl: %.2f' % val_ppl)


    def get_arch(self):
        arch, encoded_x = self.model.sample_arch()
        # self.arms[self.node_id] = Node(self.node_id, arch, self.node_id, 0)     # Returns "set"
        self.node_id += 1
        return arch

    def save(self):
        to_save = {a: self.arms[a].to_dict() for a in self.arms}
        # Only replace file if save successful so don't lose results of last pickle save
        with open(os.path.join(self.save_dir,'results_tmp.pkl'),'wb') as f:
            pickle.dump(to_save, f)
        shutil.copyfile(os.path.join(self.save_dir, 'results_tmp.pkl'), os.path.join(self.save_dir, 'results.pkl'))

        self.model.save()

    def run(self):
        while self.iters < self.B:
            arch = self.get_arch()
            self.model.train_batch(arch)
            self.iters += 1
            if self.iters % 500 == 0:
                self.save()

        self.save()

    def get_eval_arch(self, N, mask_list=[]):
        loss_list = []
        encode_list = []
        for _ in range(N):
            arch, encoded_x = self.model.sample_arch(mask_list=mask_list)     # Returns the random sampled cell
            # print("arch: ", arch)
            try:
                ppl = self.model.evaluate(arch)
            except Exception as e:
                ppl = 1000000
            # logging.info(arch)
            logging.info('objective_val: %.3f' % ppl)
            loss_list.append(ppl)
            encode_list.append(encoded_x)

        return encode_list, loss_list

def main(args):
    # Fill in with root output path
    root_dir = os.path.join(os.getcwd(), 'results')
    if args.save_dir is None:
        save_dir = os.path.join(root_dir, '%s/random/trial%d' % (args.benchmark, args.seed))
    else:
        save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info(args)

    data_size = 25000
    time_steps = 1

    mask_list = []
    learnedFeature = []
    numMeasurement = args.num_measure
    N = 196 # (2+3+4+5)*7*2

    B = int(args.epochs * data_size / args.batch_size / time_steps)  # Budget

    from darts_wrapper_discrete import DartsWrapper
    model = DartsWrapper(save_dir, args.seed, args.batch_size, args.grad_clip, args.epochs,
                         budget=B, init_channels=args.init_channels, sample_prob=args.sample_prob)

    searcher = Random_NAS(B, model, args.seed, save_dir)
    logging.info('budget: %d' % (searcher.B))
    searcher.run()                            # Trains one-shot model

    def sparse_recovery(mask_list, learnedFeature, numMeasurement):
        with open('encoded_x.pkl', 'rb') as f:
            encode_list = pickle.load(f)

        with open('measure_loss_list.pkl', 'rb') as f:
            loss_list = pickle.load(f)

        index = np.where(loss_list <= args.threshold)[0]   # Removing outliers

        print("index length: ", len(index))

        mask_list, learnedFeature = psr.PSR_lasso(x_points=encode_list[index], y_points=loss_list[index],
                                                  numMeasurement=len(index),
                                                  alpha=args.alpha, degree=2, nMono=args.num_mono, N=N, t=1,
                                                  learnedFeature=learnedFeature,
                                                  maskList=mask_list)
        mask_list = merge_masklist(mask_list=mask_list, N=N)
        print("mask_list: ", mask_list[0][0][0])
        return mask_list


    for _ in range(1):
        mask_list = sparse_recovery(mask_list=mask_list, learnedFeature=learnedFeature, numMeasurement=numMeasurement)
        mask_string = mask_to_str(mask_list)
        logging.info(mask_string)




def merge_masklist(mask_list, N):
    length = len(mask_list)

    mask = np.zeros(N)
    for i in range(length):
        minusIndex = np.where(mask_list[i][0][0] == -1.)
        mask[minusIndex] = -1.

        plusIndex = np.where(mask_list[i][0][0] == 1.)
        mask[plusIndex] = 1.

    maskList = [[(mask, 0)]]

    return maskList

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args for SHA with weight sharing')
    parser.add_argument('--benchmark', dest='benchmark', type=str, default='cnn')
    parser.add_argument('--seed', dest='seed', type=int, default=2)
    parser.add_argument('--epochs', dest='epochs', type=int, default=100)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('--grad_clip', dest='grad_clip', type=float, default=0.25)
    parser.add_argument('--save_dir', dest='save_dir', type=str, default=None)
    parser.add_argument('--init_channels', dest='init_channels', type=int, default=16)
    parser.add_argument('--sample_prob', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=10.0)
    parser.add_argument('--num_measure', type=int, default=3000)
    parser.add_argument('--num_mono', type=int, default=12)
    parser.add_argument('--threshold', type=float, default=1.5)
    args = parser.parse_args()

    main(args)
