#!/bin/bash
array=( $@ )
len=${#array[@]}
last_args=${array[@]:2:$len}

donut: --dataset donut --split val --n_inputs 100 --batch_size 1 --l1_filter maxB --l2_filter non-overlap --model_path naver-clova-ix/donut-base-finetuned-docvqa --device cuda --task next_token_pred --cache_dir ./models_cache --l1_span_thresh 0.6


bart: --dataset sst2 --split val --n_inputs 100 --batch_size 1 --l1_filter maxB --l2_filter non-overlap --model_path bert-base-uncased --device cuda --task seq_class --cache_dir ./models_cache

gpt2: --dataset sst2 --split val --n_inputs 100 --batch_size 1 --l1_filter all --l2_filter non-overlap --model_path gpt2 --device cuda --task seq_class --cache_dir ./models_cache