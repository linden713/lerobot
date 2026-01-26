#!/bin/bash
# Train Pi0.6 Star Value Function using standard LeRobot pipeline

# Activate Conda Env
rm -rf outputs/train/pi06_star_value_droid_001
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab

export WANDB_PROJECT=lerobot_pi06_star
export DATASET_REPO_ID="lerobot/droid_1.0.1"
export PYTHONPATH=src:$PYTHONPATH

# Run lerobot_train
# Using episodes=slice(0,500) as requested
# Using policy=pi06_star_value which we just implemented
# output_dir will be outputs/train/pi06_star_value_droid

python src/lerobot/scripts/lerobot_train.py \
    --policy.type=pi06_star \
    --policy.use_value_function=true \
    --dataset.repo_id=$DATASET_REPO_ID \
    --dataset.episodes="slice(0,500)" \
    --dataset.video_backend=pyav \
    --policy.device=cuda \
    --batch_size=8 \
    --steps=1000 \
    --save_freq=500 \
    --eval_freq=-1 \
    --wandb.enable=True \
    --policy.push_to_hub=False \
    --tolerance_s=0.1 \
    --output_dir=outputs/train/pi06_star_value_droid_001 \
    --job_name=pi06_star_value_droid_test

