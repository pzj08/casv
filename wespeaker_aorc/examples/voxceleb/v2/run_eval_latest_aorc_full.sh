#!/bin/bash

. ./path.sh || exit 1

set -e
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

run_root=exp/aorc_baseline_ablation_20260621_025037
exp_dir=${run_root}/aorc_full
data_type=shard
work_data=data/baseline
vox1_dir=${work_data}/vox1
train_dir=${work_data}/vox2_train_voxca
gpus="[0,1,2,3]"
num_avg=1
extract_nj=4
python_cmd=python
trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi only_ca5.kaldi only_ca10.kaldi only_ca15.kaldi only_ca20.kaldi vox_ca5.kaldi vox_ca10.kaldi vox_ca15.kaldi vox_ca20.kaldi"

. tools/parse_options.sh || exit 1

avg_model=${exp_dir}/models/avg_model.pt
echo "[aorc_full] average latest ${num_avg} checkpoint(s)"
${python_cmd} wespeaker/bin/average_model.py \
  --dst_model "${avg_model}" \
  --src_path "${exp_dir}/models" \
  --num "${num_avg}"

echo "[aorc_full] extract VoxCeleb1 embeddings"
vox1_wavs_num=$(wc -l "${vox1_dir}/wav.scp" | awk '{print $1}')
bash tools/extract_embedding.sh \
  --exp_dir "${exp_dir}" \
  --model_path "${avg_model}" \
  --data_type "${data_type}" \
  --data_list "${vox1_dir}/${data_type}.list" \
  --wavs_num "${vox1_wavs_num}" \
  --store_dir vox1 \
  --batch_size 1 \
  --num_workers 1 \
  --nj "${extract_nj}" \
  --gpus "${gpus}"

echo "[aorc_full] extract train embeddings for mean subtraction"
train_wavs_num=$(wc -l "${train_dir}/wav.scp" | awk '{print $1}')
bash tools/extract_embedding.sh \
  --exp_dir "${exp_dir}" \
  --model_path "${avg_model}" \
  --data_type "${data_type}" \
  --data_list "${train_dir}/${data_type}.list" \
  --wavs_num "${train_wavs_num}" \
  --store_dir vox2_train \
  --batch_size 1 \
  --num_workers 1 \
  --nj "${extract_nj}" \
  --gpus "${gpus}"

echo "[aorc_full] cosine scoring with mean subtraction"
rm -f "${exp_dir}/scores/baseline_cos_result"
bash local/score_baseline.sh \
  --stage 1 --stop-stage 2 \
  --exp_dir "${exp_dir}" \
  --trials "${trials}" \
  --trials_dir "${work_data}/trials" \
  --cal_mean true \
  --python_cmd "${python_cmd}"
