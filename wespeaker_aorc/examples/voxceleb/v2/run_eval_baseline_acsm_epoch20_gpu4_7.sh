#!/bin/bash

. ./path.sh || exit 1

set -e

export PATH=/xmudata/pzj/envs/casv1/bin:${PATH}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

gpus="[4,5,6,7]"
extract_nj=4
cal_mean=true
python_cmd=/xmudata/pzj/envs/casv1/bin/python

baseline_src_exp_dir=exp/baseline_resnet34_epoch20
baseline_model_path=${baseline_src_exp_dir}/models/model_20.pt
baseline_eval_dir=exp/baseline_resnet34_epoch20_eval_voxca_gpu4_7_20260628

acsm_src_exp_dir=/xmudata/pzj/casv/wespeaker_aorc/exp/ACSM-ResNet34-main-cos-work1-gpu4-7-20260628-v2
acsm_model_path=${acsm_src_exp_dir}/models/model_20.pt
acsm_eval_dir=exp/ACSM-ResNet34-main-cos-work1-gpu4-7-20260628-v2_epoch20_eval_voxca_gpu4_7_20260628

echo "[baseline epoch20] src=${baseline_src_exp_dir}"
echo "[baseline epoch20] model=${baseline_model_path}"
bash run_eval_acsm_voxca.sh \
  --src_exp_dir "${baseline_src_exp_dir}" \
  --exp_dir "${baseline_eval_dir}" \
  --model_path "${baseline_model_path}" \
  --gpus "${gpus}" \
  --extract_nj "${extract_nj}" \
  --cal_mean "${cal_mean}" \
  --python_cmd "${python_cmd}"

echo "[acsm epoch20] src=${acsm_src_exp_dir}"
echo "[acsm epoch20] model=${acsm_model_path}"
bash run_eval_acsm_voxca.sh \
  --src_exp_dir "${acsm_src_exp_dir}" \
  --exp_dir "${acsm_eval_dir}" \
  --model_path "${acsm_model_path}" \
  --gpus "${gpus}" \
  --extract_nj "${extract_nj}" \
  --cal_mean "${cal_mean}" \
  --python_cmd "${python_cmd}"

echo "[done] baseline_eval_dir=${baseline_eval_dir}"
echo "[done] acsm_eval_dir=${acsm_eval_dir}"
