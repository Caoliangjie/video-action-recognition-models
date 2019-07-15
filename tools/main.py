import os
import sys
import json
import random
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

from opts import parse_opts
from model import generate_model
from mean import get_mean, get_std
from transforms.spatial_transforms import (
    Compose, ComposeTest, Normalize, Scale, CenterCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, TestSpatialUniformCrop, RandomHorizontalFlip, ToTensor)
from transforms.temporal_transforms import LoopPadding, TemporalCenterCrop, TemporalRandomCrop, TestTemporalUniformCrop
from transforms.target_transforms import ClassLabel, VideoID, TestSegmentIndex
from transforms.target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger
from train import train_epoch
from validation import val_epoch
from test import eval_val_set, eval_test_set

if __name__ == '__main__':
    opt = parse_opts()
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    print(opt)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    if opt.manual_seed is not None:
        random.seed(opt.manual_seed)
        torch.manual_seed(opt.manual_seed)
        cudnn.deterministic = True

    model, parameters = generate_model(opt)
    print(model)
    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()

    if opt.no_mean_norm and opt.no_std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif opt.no_std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    if not opt.no_train:
        assert opt.train_crop in ['random', 'corner', 'center']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                opt.scales, opt.sample_size, crop_positions=['c'])
        if opt.no_hflip:
            spatial_transform = Compose([
                crop_method,
                ToTensor(opt.norm_value), norm_method])
        else:
            spatial_transform = Compose([
                crop_method,
                RandomHorizontalFlip(),
                ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = TemporalRandomCrop(opt.sample_duration, opt.sample_rate)
        target_transform = ClassLabel()
        training_data = get_training_set(opt, spatial_transform,
                                         temporal_transform, target_transform)
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'top1', 'top5', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'top1', "top5", 'lr'])

        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening
        optimizer = optim.SGD(
            parameters,
            lr=opt.learning_rate,
            momentum=opt.momentum,
            dampening=dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)
        assert opt.lr_policy in [0, 1]
        if opt.lr_policy == 0:
            scheduler = lr_scheduler.MultiStepLR(optimizer, [int(i) for i in opt.lr_steps.split("_")])
        elif opt.lr_policy == 1:
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=opt.lr_patience, min_lr=1e-6)

    if not opt.no_val:
        spatial_transform = Compose([
            Scale(opt.sample_size),
            CenterCrop(opt.sample_size),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = TemporalCenterCrop(opt.sample_duration, opt.sample_rate)
        target_transform = ClassLabel()
        validation_data = get_validation_set(
            opt, spatial_transform, temporal_transform, target_transform)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'top1', 'top5'])

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

    print('run')
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        # MultiStepLR
        if not opt.no_train and not opt.no_val and opt.lr_policy == 0:
            scheduler.step()

        if not opt.no_train:
            train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger)
        if not opt.no_val:
            validation_loss = val_epoch(i, val_loader, model, criterion, opt,
                                        val_logger)

        # ReduceLROnPlateau
        if not opt.no_train and not opt.no_val and opt.lr_policy == 1:
            scheduler.step(validation_loss)

    if opt.test:
        spatial_transform = ComposeTest([
            Scale(opt.sample_size),
            TestSpatialUniformCrop(opt.sample_size),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = TestTemporalUniformCrop(opt.sample_duration, opt.sample_rate)
        if opt.test_subset == "val":
            target_transform = TargetCompose([VideoID(), ClassLabel(), TestSegmentIndex()])
        elif opt.test_subset == "test":
            target_transform = TargetCompose([VideoID(), TestSegmentIndex()])

        test_data = get_test_set(opt, spatial_transform, temporal_transform,
                                 target_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)

        if opt.test_subset == "val":
            eval_val_set(test_loader, model, opt)
        elif opt.test_subset == "test":
            eval_test_set(test_loader, model, opt)
