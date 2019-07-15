import torch
import time
import sys
import json
import os
import numpy as np
import torch.nn.functional as F

from utils import AverageMeter, calculate_accuracy_and_get_pred
from evaluation import eval_val_set_results_json

def eval_val_set(data_loader, model, opt):
    model.eval()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    
    results_dict = dict()
    results_json = "results_val_set.json"
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)
            
            targets_video_id = targets[0]
            targets_label = targets[1]
            targets_test_segment_index = targets[2]
            
            if not opt.no_cuda:
                inputs = inputs.cuda(non_blocking=True)
                targets_label = targets_label.cuda(non_blocking=True)
    
            outputs = model(inputs)
            
            if not opt.no_softmax_in_test:
                outputs = F.softmax(outputs, -1)
                
            acc, pred = calculate_accuracy_and_get_pred(outputs, targets_label)
            
            accuracies.update(acc, inputs.size(0))
    
            batch_time.update(time.time() - end_time)
            end_time = time.time()
    
            print('Iter: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      acc=accuracies))
            
            targets_label = targets_label.tolist()
            targets_test_segment_index = targets_test_segment_index.tolist()
            pred = pred.tolist()[0]
            
            targets_length = len(targets_video_id)
            if not (targets_length == len(targets_label) and
                        targets_length == len(targets_test_segment_index) and 
                            targets_length == len(pred) and
                                targets_length == outputs.size()[0]):
                print("test target length not match ......")
                return
            
            for j in range(targets_length):
                target_video_id = targets_video_id[j]
                if target_video_id not in results_dict:
                    results_dict[target_video_id] = dict()
                    results_dict[target_video_id]["target_label"] = targets_label[j]
                    results_dict[target_video_id]["pred_outputs"] = [0.0] * opt.n_finetune_classes
                    
                results_dict[target_video_id]['target_label'] = targets_label[j]
                pred_outputs = results_dict[target_video_id]["pred_outputs"]
                for k in range(opt.n_finetune_classes):
                    pred_outputs[k] += outputs[j].tolist()[k]
                results_dict[target_video_id]["pred_outputs"] = pred_outputs
                
            if (i % 100) == 0:
                with open(os.path.join(opt.result_path, results_json), 'w') as f:
                    json.dump(results_dict, f)
        
        with open(os.path.join(opt.result_path, results_json), 'w') as f:
            json.dump(results_dict, f)
            
        eval_val_set_results_json(os.path.join(opt.result_path, results_json))
            
def eval_test_set(data_loader, model, opt):
    model.eval()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    
    results_dict = dict()
    results_json = "results_test_set.json"
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)
            
            targets_video_id = targets[0]
            targets_test_segment_index = targets[1]
            
            if not opt.no_cuda:
                inputs = inputs.cuda(non_blocking=True)
    
            outputs = model(inputs)
            
            if not opt.no_softmax_in_test:
                outputs = F.softmax(outputs, -1)
            
            _, pred = outputs.topk(1, 1, True)
            pred = pred.t()
                
            batch_time.update(time.time() - end_time)
            end_time = time.time()
    
            print('Iter: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time))
            
            targets_test_segment_index = targets_test_segment_index.tolist()
            pred = pred.tolist()[0]
            
            targets_length = len(targets_video_id)
            if not (targets_length == len(targets_test_segment_index) and 
                            targets_length == len(pred) and
                                targets_length == outputs.size()[0]):
                print("test target length not match ......")
                return
            
            for j in range(targets_length):
                target_video_id = targets_video_id[j]
                if target_video_id not in results_dict:
                    results_dict[target_video_id] = dict()
                    results_dict[target_video_id]["pred_outputs"] = [0.0] * opt.n_finetune_classes
                    
                pred_outputs = results_dict[target_video_id]["pred_outputs"]
                for k in range(opt.n_finetune_classes):
                    pred_outputs[k] += outputs[j].tolist()[k]
                results_dict[target_video_id]["pred_outputs"] = pred_outputs
                
            if (i % 100) == 0:
                with open(os.path.join(opt.result_path, results_json), 'w') as f:
                    json.dump(results_dict, f)
        
        with open(os.path.join(opt.result_path, results_json), 'w') as f:
            json.dump(results_dict, f)
