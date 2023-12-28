#!/bin/bash

ROOT=/mnt/lustre/chenziyu/PanoDiff
export PYTHONPATH=$ROOT:$PYTHONPATH

srun -p AD_V2X_A100_40G -n1 --quotatype=auto --gres=gpu:1 --ntasks-per-node=1 \
python tool_add_control_inpaint_9.py ./models/sd-v1-5-inpainting.ckpt ./models/control_sd15_inpainting_ini.ckpt