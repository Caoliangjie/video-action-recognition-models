#!/bin/bash

ROOT_PATH=/path/to/root

python tools/main.py \
--root_path ${ROOT_PATH} \
--video_path videos/kinetics \
--annotation_path annotations/kinetics_01.json \
--result_path results/kinetics \
--dataset kinetics --n_classes 400 \
--sample_size 112 --sample_duration 16 --sample_rate 4 \
--learning_rate 0.01 --n_epochs 100 \
--batch_size 32 --n_threads 16 --checkpoint 5  \
--model r2plus1d --model_depth 34 \
2>&1 | tee ${ROOT_PATH}/results/kinetics/log.txt
