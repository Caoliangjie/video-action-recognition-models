import os
import subprocess
import argparse
import json
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results_json',
        default='./results.json',
        type=str,
        help='Results json file path')
    parser.add_argument(
        '--test_subset',
        default='val',
        type=str,
        help='Used subset in test (val | test)')
    parser.add_argument(
        '--dataset',
        default='kinetics',
        type=str,
        help='Used dataset (kinetics | ucf101 | hmdb51 | something)')
    
    args = parser.parse_args()
    
    if args.test_subset == "val":
        eval_val_set_results_json(args.results_json)
    
def eval_val_set_results_json(results_json_path):
    
    eval_str_print = ""

    with open(results_json_path, 'r') as json_file:
        results_dict = json.load(json_file)
        
        total_count = len(results_dict)
        error_count_top1 = 0
        error_count_top5 = 0
         
        for k,v in results_dict.items():
            target_label = v['target_label']
            pred_outputs = v['pred_outputs']
            
            max_index = np.array(pred_outputs).argmax()
            if not max_index == target_label:
                error_count_top1 += 1
                
            top_indexes = np.argsort(pred_outputs)[::-1]
            hit_top5 = False
            for j in range(5):
                if top_indexes[j] == target_label:
                    hit_top5 = True
                    break
            if not hit_top5:
                error_count_top5 += 1
        
        acc_top1 = 1.0 - (error_count_top1 / total_count)
        acc_top5 = 1.0 - (error_count_top5 / total_count)
        
        eval_str_print += "\n"
        eval_str_print += " --- top1 accuracy: {}\n".format(str(acc_top1))
        eval_str_print += " --- top5 accuracy: {}\n".format(str(acc_top5))
    
    print(eval_str_print)
    