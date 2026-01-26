#!/bin/bash
#SBATCH --job-name=clip_linear_prob_on_simple_location
#SBATCH -o ./out_files/%x_%j.out
#SBATCH -e ./out_files/%x_%j.err
#SBATCH --partition=batch_vll
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=4-00:00:00
#SBATCH --mem-per-gpu=40G

python linear_prob.py \
--data_root "/local_datasets/object_direction_2D_linear_probing" \
--output_dir "results/linear_probe_simple_object_location" \
--task_name "baseline" \
--model_type "ViT-L/14" \
--cls_token