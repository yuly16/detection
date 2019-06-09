import os
import sys
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchnet as tnt
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from PIL import Image
from experiment.util import AveragePrecisionMeter, Warp


class Engine(object):
    def __init__(self, state={}):
        self.state = state
        if self._state('use_gpu') is None:
            self.state['use_gpu'] = torch.cuda.is_available()

        if self._state('image_size') is None:
            self.state['image_size'] = 224

        if self._state('batch_size') is None:
            self.state['batch_size'] = 64

        if self._state('workers') is None:
            self.state['workers'] = 4

        if self._state('evaluate') is None: 
            self.state['evaluate'] = False

        if self._state('start_epoch') is None:
            self.state['start_epoch'] = 0

        if self._state('max_epochs') is None:
            self.state['max_epochs'] = 90

        if self._state('epoch_step') is None:
            self.state['epoch_step'] = [10]

        # meters
        self.state['meter_loss'] = tnt.meter.AverageValueMeter()
        # time measure
        self.state['batch_time'] = tnt.meter.AverageValueMeter()
        self.state['data_time'] = tnt.meter.AverageValueMeter()
        # display parameters
        if self._state('use_pb') is None:
            self.state['use_pb'] = True

    def _state(self, name):
        if name in self.state:
            return self.state[name]

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None):
        self.state['meter_loss'].reset()
        self.state['batch_time'].reset()
        self.state['data_time'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None):
        loss = self.state['meter_loss'].value()[0]
        return loss

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None):
        pass

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None):

        # record loss
        self.state['loss_batch'] = self.state['loss'].data[0]
        self.state['meter_loss'].add(self.state['loss_batch'])

    def on_forward(self, training, model, criterion, data_loader, optimizer=None):
        input_var = torch.autograd.Variable(self.state['input'])
        target_var = torch.autograd.Variable(self.state['target'])

        if not training:
            input_var.volatile = True
            target_var.volatile = True

        # compute output
        self.state['output'] = model(input_var)
        self.state['loss'] = criterion(self.state['output'], target_var)

        if training:
            optimizer.zero_grad()
            self.state['loss'].backward()
            optimizer.step()

    def multi_learning(self, model, criterion, train_dataset, val_dataset):
        src_state_dict = deepcopy(model.state_dict())
        src_state = deepcopy(self.state)
        scales = self.state['image_size'].split(',')
        state_dicts = []
        if not os.path.exists(self.state['save_model_path']):
            os.makedirs(self.state['save_model_path'])
        filename = os.path.join(self.state['save_model_path'], 'model.pth.tar')
        for imsize in scales:
            self.state = deepcopy(src_state)
            self.state['image_size'] = int(imsize)
            model.load_state_dict(src_state_dict)
            optimizer = torch.optim.SGD(model.parameters(), lr=self.state['lr'], momentum=self.state['momentum'], weight_decay=self.state['weight_decay'])
            state_dicts.append(self.learning(model, criterion, train_dataset, val_dataset, optimizer)) 
            print('save model {filename}'.format(filename=filename))
            torch.save(state_dicts, filename)

    def init_learning(self, model, criterion):
        if self._state('train_transform') is None:
            self.state['train_transform'] = transforms.Compose([
                Warp(self.state['image_size'] + 30),
                transforms.RandomCrop(self.state['image_size']),
                transforms.RandomHorizontalFlip(),
                lambda x: torch.from_numpy(np.array(x)).permute(2, 0, 1).float(),
                lambda x: x.index_select(0, torch.LongTensor([2,1,0])),
                lambda x: x - torch.Tensor(model.image_normalization_mean).view(3, 1, 1),
            ])

        if self._state('val_transform') is None:
            self.state['val_transform'] = transforms.Compose([
                Warp(self.state['image_size']),
                lambda x: torch.from_numpy(np.array(x)).permute(2, 0, 1).float(),
                lambda x: x.index_select(0, torch.LongTensor([2,1,0])),
                lambda x: x - torch.Tensor(model.image_normalization_mean).view(3, 1, 1),
            ])

        self.state['best_score'] = 0

    def learning(self, model, criterion, train_dataset, val_dataset, optimizer=None):

        self.init_learning(model, criterion)
        
        # define train and val transform
        train_dataset.transform = self.state['train_transform']
        train_dataset.target_transform = self._state('train_target_transform')
        val_dataset.transform = self.state['val_transform']
        val_dataset.target_transform = self._state('val_target_transform')

        # data loader
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.state['batch_size'], shuffle=True,
                                                   num_workers=self.state['workers'])

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=self.state['batch_size'], shuffle=False,
                                                 num_workers=self.state['workers'])

        # optionally resume from a checkpoint
        if self._state('resume') is not None:
            if os.path.isfile(self.state['resume']):
                print("=> loading checkpoint '{}'".format(self.state['resume']))
                checkpoint = torch.load(self.state['resume'])
                self.state['start_epoch'] = checkpoint['epoch']
                self.state['best_score'] = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.state['evaluate'], checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.state['resume']))


        if self.state['use_gpu']:
            train_loader.pin_memory = True
            val_loader.pin_memory = True
            cudnn.benchmark = True

            model = torch.nn.DataParallel(model).cuda()
            criterion = criterion.cuda()

        if self.state['evaluate']:
            self.validate(val_loader, model, criterion)
            return

        # TODO define optimizer
        for epoch in range(self.state['start_epoch'], self.state['max_epochs']):
            self.state['epoch'] = epoch
            self.adjust_learning_rate(optimizer)

            # train for one epoch
            self.train(train_loader, model, criterion, optimizer, epoch)
            # evaluate on validation set
            prec1 = self.validate(val_loader, model, criterion)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > self.state['best_score']
            self.state['best_score'] = max(prec1, self.state['best_score'])
            print(' *** best={best:.3f}'.format(best=self.state['best_score']))

        cur_state = {
                'epoch': epoch + 1,
                'image_size': self.state['image_size'],
                'arch': self._state('arch'),
                'state_dict': model.module.state_dict() if self.state['use_gpu'] else model.state_dict(),
                'best_score': self.state['best_score'],
            }
        # filename = os.path.join(self.state['save_model_path'], 'checkpoint_{}.pth.tar'.format(self.state['image_size']))
        # torch.save(deepcopy(cur_state), filename)
        return deepcopy(cur_state)

    def train(self, data_loader, model, criterion, optimizer, epoch):
        # switch to train mode
        model.train()
        self.on_start_epoch(True, model, criterion, data_loader, optimizer)
        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Training')
        end = time.time()
        for i, (input, target) in enumerate(data_loader):
            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])
            

            self.state['input'] = input
            self.state['target'] = target

            self.on_start_batch(True, model, criterion, data_loader, optimizer)
            if self.state['use_gpu']:
                self.state['target'] = self.state['target'].cuda(async=True)

            self.on_forward(True, model, criterion, data_loader, optimizer)
            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch(True, model, criterion, data_loader, optimizer)
        self.on_end_epoch(True, model, criterion, data_loader, optimizer)

    def validate(self, data_loader, model, criterion):

        # switch to evaluate mode
        model.eval()

        self.on_start_epoch(False, model, criterion, data_loader)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Test')

        end = time.time()
        for i, (input, target) in enumerate(data_loader):
            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input'] = input
            self.state['target'] = target

            self.on_start_batch(False, model, criterion, data_loader)

            if self.state['use_gpu']:
                self.state['target'] = self.state['target'].cuda(async=True)

            self.on_forward(False, model, criterion, data_loader)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch(False, model, criterion, data_loader)

        score = self.on_end_epoch(False, model, criterion, data_loader)

        return score

    def adjust_learning_rate(self, optimizer):
        """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
        # lr = args.lr * (0.1 ** (epoch // 10))
        if self.state['epoch'] is not 0 and self.state['epoch'] in self.state['epoch_step']:
            print('update learning rate')
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
                print(param_group['lr'])

class MultiLabelMAPEngine(Engine):
    def __init__(self, state):
        Engine.__init__(self, state)
        if self._state('difficult_examples') is None:
            self.state['difficult_examples'] = False
        self.state['ap_meter'] = AveragePrecisionMeter(self.state['difficult_examples'])

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None):
        Engine.on_start_epoch(self, training, model, criterion, data_loader, optimizer)
        self.state['ap_meter'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None):
        map = 100 * self.state['ap_meter'].value().mean()
        loss = self.state['meter_loss'].value()[0]
        if training:
            strs = 'Scale: {imsize}\t Epoch: [{epoch}]\t trainLoss {loss:.4f}\t trainMAP {map:.3f}'.format(imsize=self.state['image_size'], epoch=self.state['epoch'], loss=loss, map=map)
        else:
            strs = 'testLoss {loss:.4f}\t testMAP {map:.3f}'.format(loss=loss, map=map)
        print(strs)
        return map

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None):

        self.state['target_gt'] = self.state['target'].clone()
        self.state['target'][self.state['target'] == -1] = 0

        input = self.state['input']
        self.state['input'] = input
        # self.state['name'] = input[1]

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None):

        Engine.on_end_batch(self, training, model, criterion, data_loader, optimizer)
        # measure mAP
        self.state['ap_meter'].add(self.state['output'].data, self.state['target_gt'])