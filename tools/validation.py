import torch
import time
import random
import sys

from utils import AverageMeter, calculate_accuracy_topk


def val_epoch(epoch, data_loader, model, criterion, opt, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()

    end_time = time.time()
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            if not opt.no_cuda:
                inputs = inputs.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            top1, top5 = calculate_accuracy_topk(outputs, targets, topk=(1,5))

            losses.update(loss.item(), inputs.size(0))
            acc_top1.update(top1, inputs.size(0))
            acc_top5.update(top5, inputs.size(0))
    
            batch_time.update(time.time() - end_time)
            end_time = time.time()
    
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top1 {acc_top1.val:.3f} ({acc_top1.avg:.3f})\t'
                  'Top5 {acc_top5.val:.3f} ({acc_top5.avg:.3f})'.format(
                      epoch,
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      acc_top1=acc_top1,
                      acc_top5=acc_top5))

    logger.log({'epoch': epoch, 'loss': losses.avg, 'top1': acc_top1.avg, 'top5': acc_top5.avg})

    return losses.avg
