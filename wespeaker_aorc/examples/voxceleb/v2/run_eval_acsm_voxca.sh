#!/bin/bash

. ./path.sh || exit 1

set -e

export PATH=/xmudata/pzj/envs/casv1/bin:${PATH}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

src_exp_dir=exp/ACSM-ResNet34-TSTP-emb256-fbank80
exp_dir=
model_path=
avg_num=10
min_epoch=0
max_epoch=65536
data_type=shard
work_data=data/baseline
vox1_dir=${work_data}/vox1
train_dir=${work_data}/vox2_train_voxca
gpus="[0,1,2,3]"
extract_nj=4
batch_size=1
num_workers=1
cal_mean=true
python_cmd=/xmudata/pzj/envs/casv1/bin/python
trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi only_ca5.kaldi only_ca10.kaldi only_ca15.kaldi only_ca20.kaldi vox_ca5.kaldi vox_ca10.kaldi vox_ca15.kaldi vox_ca20.kaldi"

. tools/parse_options.sh

if [ -z "${exp_dir}" ]; then
  exp_dir=${src_exp_dir}_eval_voxca
fi
if [ ! -f "${src_exp_dir}/config.yaml" ]; then
  echo "Missing source config: ${src_exp_dir}/config.yaml" >&2
  exit 1
fi

mkdir -p "${exp_dir}/models"
cp "${src_exp_dir}/config.yaml" "${exp_dir}/config.yaml"

if [ -z "${model_path}" ]; then
  model_path=${exp_dir}/models/avg_model.pt
  echo "[ACSM Vox-CA eval] average last ${avg_num} checkpoints from ${src_exp_dir}/models"
  "${python_cmd}" wespeaker/bin/average_model.py \
    --dst_model "${model_path}" \
    --src_path "${src_exp_dir}/models" \
    --num "${avg_num}" \
    --min_epoch "${min_epoch}" \
    --max_epoch "${max_epoch}"
else
  echo "[ACSM Vox-CA eval] use model_path=${model_path}"
fi

echo "[ACSM Vox-CA eval] extract VoxCeleb1 test embeddings"
vox1_wavs_num=$(wc -l "${vox1_dir}/wav.scp" | awk '{print $1}')
bash tools/extract_embedding.sh \
  --exp_dir "${exp_dir}" \
  --model_path "${model_path}" \
  --data_type "${data_type}" \
  --data_list "${vox1_dir}/${data_type}.list" \
  --wavs_num "${vox1_wavs_num}" \
  --store_dir vox1 \
  --batch_size "${batch_size}" \
  --num_workers "${num_workers}" \
  --nj "${extract_nj}" \
  --gpus "${gpus}"

if [ "${cal_mean}" = "true" ] || [ "${cal_mean}" = "True" ] || [ "${cal_mean}" = "1" ]; then
  echo "[ACSM Vox-CA eval] extract Vox-CA train embeddings for mean subtraction only"
  train_wavs_num=$(wc -l "${train_dir}/wav.scp" | awk '{print $1}')
  bash tools/extract_embedding.sh \
    --exp_dir "${exp_dir}" \
    --model_path "${model_path}" \
    --data_type "${data_type}" \
    --data_list "${train_dir}/${data_type}.list" \
    --wavs_num "${train_wavs_num}" \
    --store_dir vox2_train \
    --batch_size "${batch_size}" \
    --num_workers "${num_workers}" \
    --nj "${extract_nj}" \
    --gpus "${gpus}"
fi

echo "[ACSM Vox-CA eval] cosine scoring on MIM/Vox-CA test sets"
rm -f "${exp_dir}/scores/baseline_cos_result"
bash local/score_baseline.sh \
  --stage 1 --stop-stage 2 \
  --exp_dir "${exp_dir}" \
  --trials "${trials}" \
  --trials_dir "${work_data}/trials" \
  --cal_mean "${cal_mean}" \
  --python_cmd "${python_cmd}"
