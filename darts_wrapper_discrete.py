import sys
import genotypes
from models.search_cnn import SearchCNN
import utils

import time
import math
import copy
import random
import logging
import os
import gc
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import utils

device = torch.device("cuda")

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class DartsWrapper:
    def  __init__(self, save_path, seed, batch_size, grad_clip, epochs, budget, resume_iter=None, init_channels=16, sample_prob=0.5):
        args = {}
        args['data'] = os.path.join(os.getcwd(), 'data')
        args['epochs'] = epochs
        args['learning_rate'] = 0.025
        args['batch_size'] = batch_size
        args['learning_rate_min'] = 0.001
        args['momentum'] = 0.9
        args['weight_decay'] = 3e-4
        args['init_channels'] = init_channels
        args['layers'] = 8
        args['drop_path_prob'] = 0.3
        args['grad_clip'] = grad_clip
        args['train_portion'] = 0.5
        args['seed'] = seed
        args['log_interval'] = 50
        args['save'] = save_path
        args['gpu'] = 0
        args['cuda'] = True
        args['cutout'] = False
        args['cutout_length'] = 16
        args['report_freq'] = 50
        args['dataset'] = 'cifar10'
        args = AttrDict(args)
        self.args = args
        self.seed = seed
        self.budget = budget
        self.save_dir = save_path
        self.sample_prob = sample_prob

        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = False
        cudnn.enabled=True
        cudnn.deterministic=True
        torch.cuda.manual_seed_all(args.seed)


        input_size, input_channels, n_classes, train_data = utils.get_data(
            args.dataset, args.data, cutout_length=0, validation=False)

        net_crit = nn.CrossEntropyLoss().to(device)
        self.net_crit = net_crit
        model = SearchCNN(input_channels, args.init_channels, n_classes, args.layers, net_crit)
        model = model.to(device)
        self.model = model
        # weights optimizer
        w_optim = torch.optim.SGD(model.weights(), args.learning_rate, momentum=args.momentum,
                                  weight_decay=args.weight_decay)

        # split data to train/validation
        n_train = len(train_data)
        split = 25000
        indices = list(range(n_train))
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
        self.train_queue = torch.utils.data.DataLoader(train_data,
                                                   batch_size=args.batch_size,
                                                   sampler=train_sampler,
                                                   num_workers=0,
                                                   pin_memory=True,
                                                   worker_init_fn=np.random.seed(args.seed))
        self.valid_queue = torch.utils.data.DataLoader(train_data,
                                                   batch_size=args.batch_size,
                                                   sampler=valid_sampler,
                                                   num_workers=0,
                                                   pin_memory=True,
                                                   worker_init_fn=np.random.seed(args.seed))



        self.train_iter = iter(self.train_queue)
        self.valid_iter = iter(self.valid_queue)

        self.steps = 0
        self.epochs = 0
        self.total_loss = 0
        self.start_time = time.time()


        try:
            self.load()
            logging.info('loaded previously saved weights')
        except Exception as e:
            print(e)

        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

        optimizer = torch.optim.SGD(
          self.model.parameters(),
          args.learning_rate,
          momentum=args.momentum,
          weight_decay=args.weight_decay)
        self.optimizer = optimizer

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
          optimizer, float(args.epochs), eta_min=args.learning_rate_min)

        if resume_iter is not None:
            self.steps = resume_iter
            self.epochs = int(
                resume_iter / len(self.train_queue))
            logging.info("Resuming from epoch %d" % self.epochs)
            self.objs = utils.AvgrageMeter()
            self.top1 = utils.AvgrageMeter()
            self.top5 = utils.AvgrageMeter()
            for i in range(self.epochs):
                self.scheduler.step()

        size = 0
        for p in model.parameters():
            size += p.nelement()
        logging.info('param size: {}'.format(size))

        total_params = sum(x.data.nelement() for x in model.parameters())
        logging.info('Args: {}'.format(args))
        logging.info('Model total parameters: {}'.format(total_params))

    def train_batch(self, arch):
      args = self.args
      if self.steps % len(self.train_queue) == 0:
        self.scheduler.step()
        self.objs = utils.AvgrageMeter()
        self.top1 = utils.AvgrageMeter()
        self.top5 = utils.AvgrageMeter()
      lr = self.scheduler.get_lr()[0]

      self.get_weights_from_arch(arch)
      # self.set_model_weights(weights)

      step = self.steps % len(self.train_queue)
      input, target = next(self.train_iter)

      self.model.cuda()
      self.model.train()
      n = input.size(0)

      input = Variable(input, requires_grad=False).cuda()
      target = Variable(target, requires_grad=False).cuda()

      # get a random minibatch from the search queue with replacement
      self.optimizer.zero_grad()
      logits = self.model(input)
      loss = self.net_crit(logits, target)

      loss.backward()
      nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)
      self.optimizer.step()

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      self.objs.update(loss.item(), n)
      self.top1.update(prec1.item(), n)
      self.top5.update(prec5.item(), n)

      if step % args.report_freq == 0:
        logging.info('train %03d %e %f %f', step, self.objs.avg, self.top1.avg, self.top5.avg)

      self.steps += 1
      if self.steps % len(self.train_queue) == 0:
        self.epochs += 1
        self.train_iter = iter(self.train_queue)
        valid_loss = self.evaluate(arch)
        logging.info('epoch %d  |  train_acc %f  |  valid_loss %f' % (self.epochs, self.top1.avg, valid_loss))
        self.save()                 # Saving the weights



    def evaluate(self, arch, split=None):
      # Return error since we want to minimize obj val
      # logging.info(arch)
      objs = utils.AvgrageMeter()
      top1 = utils.AvgrageMeter()
      top5 = utils.AvgrageMeter()

      self.get_weights_from_arch(arch)
      # self.set_model_weights(weights)

      self.model.cuda()
      self.model.eval()

      if split is None:
        n_batches = 10
      else:
        n_batches = len(self.valid_queue)

      for step in range(n_batches):
        try:
          input, target = next(self.valid_iter)
        except Exception as e:
          logging.info('looping back over valid set')
          self.valid_iter = iter(self.valid_queue)
          input, target = next(self.valid_iter)

        with torch.no_grad():
            input = Variable(input).cuda()
            target = Variable(target).cuda()

            logits = self.model(input)
            loss = self.net_crit(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            # if step % self.args.report_freq == 0:
            #   logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

      return objs.avg

    def save(self):
        utils.save(self.model, os.path.join(self.args.save, 'weights.pt'))

    def load(self):
        utils.load(self.model, os.path.join(self.args.save, 'weights.pt'))

    def get_weights_from_arch(self, arch):
        k = sum(1 for i in range(self.model._steps) for n in range(2+i))
        num_ops = len(genotypes.PRIMITIVES)
        n_nodes = self.model._steps

        indexBuffer = 0
        for i in range(self.model._steps):
            self.model.alpha_normal[i] = nn.Parameter(arch[indexBuffer:indexBuffer+i+2])
            indexBuffer = indexBuffer+i+2

        for i in range(self.model._steps):
            self.model.alpha_reduce[i] = nn.Parameter(arch[indexBuffer:indexBuffer + i + 2])
            indexBuffer = indexBuffer + i + 2

    def set_model_weights(self, weights):
      self.model.alphas_normal = weights[0]
      self.model.alphas_reduce = weights[1]
      self.model._arch_parameters = [self.model.alphas_normal, self.model.alphas_reduce]

    def sample_arch(self, mask_list=[]):
        k = sum(1 for i in range(self.model._steps) for n in range(2+i))    # self.model._steps = 4
        num_ops = len(genotypes.PRIMITIVES)
        n_nodes = self.model._steps

        # initProb = 1/2
        # initProb = 1/8
        # sample_prob = 1/2
        current_x = torch.zeros(2*k, num_ops)
        encoded_x = np.zeros(2*k*num_ops)



        if mask_list == []:
          mask = np.zeros(2*k*num_ops)
            # sampleProb = initProb
            # current_x = torch.tensor([1]*(k*num_ops) + [-1]*(k*num_ops))
            # idx = torch.randperm(current_x.nelement())
            # current_x = current_x.view(-1)[idx].view(current_x.size())
            # current_x = current_x.reshape((-1, num_ops))
            #
            # encoded_x = np.array([1]*(k*num_ops) + [-1]*(k*num_ops))
            # np.random.shuffle(encoded_x)

        else:
            mask = mask_list[0][0][0]               # [mask][picked][encode or value]
            # posLength = len(np.where(mask == 1.)[0])
            # negLength = len(np.where(mask == -1.)[0])
            # totalLength = posLength + negLength
            # sampleProb = (2*k*num_ops*initProb - posLength) / (2*k*num_ops - totalLength)   # sampleProb allows to
                                                                                            # activate n*initProb edges
            # sampleProb = 1/2

        for i in range(2 * k):
            for j in range(num_ops):
                if (mask[num_ops * i + j] == 1):
                    encoded_x[num_ops * i + j] = 1
                    current_x[i][j] = 1
                elif (mask[num_ops * i + j] == -1):
                    encoded_x[num_ops * i + j] = -1
                    current_x[i][j] = -1
                elif (mask[num_ops * i + j] == 0):
                    buff = np.random.binomial(1, self.sample_prob)
                    encoded_x[num_ops * i + j] = 1 if buff == 1 else -1
                    current_x[i][j] = buff

        return current_x, encoded_x

    def perturb_arch(self, arch):
        new_arch = copy.deepcopy(arch)
        num_ops = len(genotypes.PRIMITIVES)

        cell_ind = np.random.choice(2)
        step_ind = np.random.choice(self.model._steps)
        nodes_in = np.random.choice(step_ind+2, 2, replace=False)
        ops = np.random.choice(range(num_ops), 2)

        new_arch[cell_ind][2*step_ind] = (nodes_in[0], ops[0])
        new_arch[cell_ind][2*step_ind+1] = (nodes_in[1], ops[1])
        return new_arch


