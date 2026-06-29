#!/bin/bash

set -euo pipefail

cd /xmudata/pzj/casv/wespeaker_acsm

export PATH="/xmudata/pzj/casv/wespeaker_acsm/examples/voxceleb/v2:${PATH}"
export PYTHONIOENCODING=UTF-8
export PYTHONPATH="/xmudata/pzj/casv/wespeaker_acsm:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

config=examples/voxceleb/v2/conf/resnet34_acsm_main_v3.yaml
exp_dir=exp/ACSM-ResNet34-main-v3-cos-work1-gpu4-7-20260629
data_dir=examples/voxceleb/v2/data/baseline/vox2_train_voxca

/xmudata/pzj/envs/casv1/bin/torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  wespeaker/bin/train.py \
    --config "${config}" \
    --exp_dir "${exp_dir}" \
    --gpus "[0,1,2,3]" \
    --num_avg 10 \
    --data_type shard \
    --train_data "${data_dir}/shard_work1_full.list" \
    --train_label "${data_dir}/utt2spk" \
    --key_filter_file "${data_dir}/key.list"
