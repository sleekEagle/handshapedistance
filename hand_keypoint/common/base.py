# Copyright (c) 2020 Graz University of Technology All rights reserved.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
import os.path as osp
import glob
import abc
import torch.optim

from hand_keypoint.config import cfg

from hand_keypoint.common.timer import Timer
from hand_keypoint.common.logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
from hand_keypoint.model import get_model

class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name='logs.txt'):
        
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return


class Trainer(Base):
    
    def __init__(self):
        super(Trainer, self).__init__(log_name = 'train_logs.txt')

    def get_optimizer(self, model):
        # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if ("backbone_net" not in n and 'decoder_net' not in n) and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if ("backbone_net" in n or 'decoder_net' in n) and p.requires_grad],
                "lr": cfg.lr*0.1,
            },
        ]


        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.lr_drop, gamma=1/cfg.lr_dec_factor)
        return optimizer, lr_scheduler

    def set_lr(self, epoch):
        if len(cfg.lr_dec_epoch) == 0:
            return cfg.lr

        for e in cfg.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < cfg.lr_dec_epoch[-1]:
            idx = cfg.lr_dec_epoch.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** len(cfg.lr_dec_epoch))



    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = get_model('train', self.joint_num)
        print('Number of trainable parameters = %d' % (self.count_parameters(model)))
        model = model.cuda()
        model = DataParallel(model)

        optimizer, lr_scheduler = self.get_optimizer(model)
        if cfg.continue_train:
            start_epoch, model, optimizer, lr_scheduler = self.load_model(model, optimizer, lr_scheduler)
        else:
            start_epoch = 0
        model.train()
        
        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
         
    def save_model(self, state, epoch, iter=0):
        file_path = osp.join(cfg.model_dir,'snapshot_%s_%s.pth.tar'%(str(epoch), str(iter)))
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_stage1(self, model):
        model_path = cfg.preload_model_path
        self.logger.info('Load stage1 checkpoint from {}'.format(model_path))
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'], strict=False)
        return model

    def load_model(self, model, optimizer, lr_scheduler):
        model_file_list = glob.glob(osp.join(cfg.model_dir,'*.pth.tar'))
        model_file_list = [file_name[file_name.find('snapshot_'):] for file_name in model_file_list]
        cur_epoch = max([int(file_name.split('_')[1].split('.')[0]) for file_name in
                         model_file_list if 'snapshot' in file_name])
        cur_iter = max([int(file_name.split('_')[2].split('.')[0]) for file_name in
                         model_file_list if 'snapshot_%d'%cur_epoch in file_name])
        model_path = osp.join(cfg.model_dir, 'snapshot_' + str(cur_epoch) + '_' + str(cur_iter) + '.pth.tar')

        self.logger.info('Load checkpoint from {}'.format(model_path))
        ckpt = torch.load(model_path)
        start_epoch = ckpt['epoch'] + 1

        resnet_dec_keys = []
        resnet_new_keys = []
        for k in ckpt['network'].keys():
            if 'decoder_net' in k and 'resnet_decoder' not in k:
                new_k = k[:19] + 'resnet_decoder.' + k[19:]
                resnet_new_keys.append(new_k)
                resnet_dec_keys.append(k)
        for i, k in enumerate(resnet_dec_keys):
            ckpt['network'][resnet_new_keys[i]] = ckpt['network'][resnet_dec_keys[i]]
            del ckpt['network'][k]

        model.load_state_dict(ckpt['network'], strict=False)
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
        except:
            pass

        try:
            lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        except:
            pass

        return start_epoch, model, optimizer, lr_scheduler


class Tester(Base):
    
    def __init__(self, ckpt_path):
        self.ckpt_path = ckpt_path
        super(Tester, self).__init__(log_name = 'test_logs.txt')


    def _make_model(self):
        model_path = self.ckpt_path
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))
        
        # prepare network
        self.logger.info("Creating graph...")
        joint_num=21
        model = get_model('test', joint_num)
        model = model.cuda()
        model = DataParallel(model)
        ckpt = torch.load(model_path)

        resnet_dec_keys = []
        resnet_new_keys = []
        for k in ckpt['network'].keys():
            if 'decoder_net' in k and 'resnet_decoder' not in k:
                new_k = k[:19] + 'resnet_decoder.' + k[19:]
                resnet_new_keys.append(new_k)
                resnet_dec_keys.append(k)
        for i, k in enumerate(resnet_dec_keys):
            ckpt['network'][resnet_new_keys[i]] = ckpt['network'][resnet_dec_keys[i]]
            del ckpt['network'][k]

        model.load_state_dict(ckpt['network'], strict=True)
        model.eval()

        self.model = model

    def _evaluate(self, preds,gt, ckpt_path, annot_subset):
        if cfg.dataset == 'InterHand2.6M':
            if cfg.predict_2p5d and cfg.predict_type == 'vectors':
                self.testset.evaluate_2p5d(preds, gt, ckpt_path, annot_subset)
            else:
                self.testset.evaluate(preds, gt, ckpt_path, annot_subset
                                      )
        elif cfg.dataset == 'ho3d':
            self.testset.evaluate(preds, ckpt_path, gt)
        elif cfg.dataset == 'h2o3d':
            self.testset.evaluate(preds, ckpt_path, gt)

    def _dump_results(self, preds, dump_dir):
        self.testset.dump_results(preds, dump_dir)

