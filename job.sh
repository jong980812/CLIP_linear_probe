#!/bin/bash
#SBATCH --job-name=clip_linear_baseline_layer3oom_addressing
#SBATCH -o ./out_files/%x_%j.out
#SBATCH -e ./out_files/%x_%j.err
#SBATCH --partition=batch_vll
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=4-00:00:00
#SBATCH --mem-per-gpu=100G

python linear_prob.py \
--data_root "/local_datasets/object_direction_2D_linear_probing" \
--output_dir "results/linear_probe_simple_object_location" \
--task_name "layer3_cls_oom_addressing" \
--model_type "ViT-L/14" \
--cls_token \
--layer_idx 3 \
