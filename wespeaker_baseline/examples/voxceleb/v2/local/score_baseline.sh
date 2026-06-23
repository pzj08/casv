#!/bin/bash

# Copyright 2026
# Licensed under the Apache License, Version 2.0.

exp_dir=
trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi only_ca5.kaldi only_ca10.kaldi only_ca15.kaldi only_ca20.kaldi vox_ca5.kaldi vox_ca10.kaldi vox_ca15.kaldi vox_ca20.kaldi"
trials_dir=data/baseline/trials
cal_mean=true
cal_mean_dir=
eval_scp_path=
ptarget=0.01
cfa=1
cmiss=1
python_cmd=python
stage=-1
stop_stage=-1

. tools/parse_options.sh
. path.sh
set -e

if [ -z "${exp_dir}" ]; then
  echo "score_baseline.sh requires --exp_dir" >&2
  exit 1
fi
if [ -z "${cal_mean_dir}" ]; then
  cal_mean_dir=${exp_dir}/embeddings/vox2_train
fi
if [ -z "${eval_scp_path}" ]; then
  eval_scp_path=${exp_dir}/embeddings/vox1/xvector.scp
fi
case "${cal_mean}" in
  true|True|1) cal_mean_py=True ;;
  false|False|0) cal_mean_py=False ;;
  *)
    echo "Unsupported cal_mean value: ${cal_mean}" >&2
    exit 1
    ;;
esac

mkdir -p ${exp_dir}/scores
scores_dir=${exp_dir}/scores

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Apply cosine scoring only."
  echo "backend scoring switches: plda=false asnorm=false snorm=false qmf=false calibration=false score_fusion=false lmft=false"
  echo "mean_subtraction=${cal_mean} cal_mean_dir=${cal_mean_dir}"
  for x in $trials; do
    echo $x
    ${python_cmd} wespeaker/bin/score.py \
      --exp_dir ${exp_dir} \
      --eval_scp_path ${eval_scp_path} \
      --cal_mean=${cal_mean_py} \
      --cal_mean_dir ${cal_mean_dir} \
      ${trials_dir}/${x}
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Compute EER/minDCF."
  echo "minDCF p_target=${ptarget} c_fa=${cfa} c_miss=${cmiss}"
  for x in $trials; do
    ${python_cmd} wespeaker/bin/compute_metrics.py \
      --p_target ${ptarget} \
      --c_fa ${cfa} \
      --c_miss ${cmiss} \
      ${scores_dir}/${x}.score \
      2>&1 | tee -a ${scores_dir}/baseline_cos_result
  done
fi
