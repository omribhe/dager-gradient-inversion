#!/bin/bash
array=( $@ )
len=${#array[@]}
last_args=${array[@]:2:$len}

echo "Creating dir"
mkdir -p /scratch/ivo_petrov/models_cache
echo "Cpying files"
rsync -a --exclude finetune/ "$(rsync-path /home/ivo_petrov)"/dager/ /scratch/ivo_petrov/dager
echo "Moving to dir"
cd /scratch/ivo_petrov/dager
echo "Running..."
python attack.py --dataset $1 --split val --n_inputs 100 --batch_size $2 --l1_filter all --l2_filter non-overlap --model_path openai-community/gpt2-large --device cuda --task seq_class --cache_dir ./models_cache $last_args
