#!/bin/bash
# Train Pi0.6 Star Value Function using standard LeRobot pipeline

# Activate Conda Env
rm -rf outputs/train/pi06_star_value_roboarena
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab

export WANDB_PROJECT=lerobot_pi06_star
export DATASET_REPO_ID="/home/lch/lerobot/roboarena_converted"
export PYTHONPATH=src:$PYTHONPATH

# Run lerobot_train
# Using episodes=slice(0,500) as requested
# Using policy=pi06_star_value which we just implemented
# output_dir will be outputs/train/pi06_star_value_droid

python src/lerobot/scripts/lerobot_train.py \
    --policy.type=pi06_star \
    --policy.use_value_function=true \
    --dataset.repo_id=$DATASET_REPO_ID \
    --num_workers=12 \
    --policy.device=cuda \
    --policy.paligemma_variant=gemma_670m \
    --batch_size=4 \
    --steps=20000 \
    --save_freq=5000 \
    --eval_freq=5000 \
    --wandb.enable=True \
    --policy.push_to_hub=False \
    --tolerance_s=0.1 \
    --output_dir=outputs/train/pi06_star_value_roboarena \
    --job_name=pi06_star_value_roboarena_test

