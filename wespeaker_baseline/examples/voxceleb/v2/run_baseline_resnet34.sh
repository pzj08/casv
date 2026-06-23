#!/bin/bash

# Copyright 2026
# Licensed under the Apache License, Version 2.0.

. ./path.sh || exit 1

stage=0
stop_stage=0

HOST_NODE_ADDR="localhost:29400"
num_nodes=1
job_id=2606

data=/data
work_data=data/baseline
data_type=shard

config=conf/baseline_resnet34.yaml
exp_dir=exp/Baseline-ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin48-SGD-epoch150
gpus="[0,1,2,3]"
num_avg=10
checkpoint=

train_label=/data/vox2_dev/utt2spk
age_label_file=/xmudata/pzj/vox-ca/vox2dev/segment2age.npy
vox2_wav_scp=/data/vox2_dev/wav.scp
vox1_wav_scp=/data/vox1/wav.scp
musan_wav_scp=/data/musan/wav.scp
rir_wav_scp=/data/rirs/wav.scp
musan_lmdb=/data/musan/lmdb
rir_lmdb=/data/rirs/lmdb

num_utts_per_shard=1000
num_threads=16
extract_nj=4
python_cmd=python
torchrun_cmd=torchrun

trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi only_ca5.kaldi only_ca10.kaldi only_ca15.kaldi only_ca20.kaldi vox_ca5.kaldi vox_ca10.kaldi vox_ca15.kaldi vox_ca20.kaldi"
use_mean_subtraction=true
ptarget=0.01
cfa=1
cmiss=1

. tools/parse_options.sh || exit 1
set -e
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

train_dir=${work_data}/vox2_train_voxca
vox1_dir=${work_data}/vox1

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Prepare baseline data views."
  ${python_cmd} local/prepare_baseline_data.py \
    --out_dir ${work_data} \
    --train_dir_name vox2_train_voxca \
    --train_label ${train_label} \
    --age_label_file ${age_label_file} \
    --vox2_wav_scp ${vox2_wav_scp} \
    --vox1_wav_scp ${vox1_wav_scp}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Build baseline train shards and VoxCeleb1 extraction shards."
  mkdir -p ${train_dir}/shards ${vox1_dir}/shards
  ${python_cmd} tools/make_shard_list.py --num_utts_per_shard ${num_utts_per_shard} \
    --num_threads ${num_threads} \
    --prefix shards \
    --shuffle \
    ${train_dir}/wav.scp ${train_dir}/utt2spk \
    ${train_dir}/shards ${train_dir}/shard.list
  cp ${vox1_wav_scp} ${vox1_dir}/wav.scp
  awk '{print $1, $1}' ${vox1_dir}/wav.scp > ${vox1_dir}/utt2spk
  ${python_cmd} tools/make_shard_list.py --num_utts_per_shard ${num_utts_per_shard} \
    --num_threads ${num_threads} \
    --prefix shards \
    ${vox1_dir}/wav.scp ${vox1_dir}/utt2spk \
    ${vox1_dir}/shards ${vox1_dir}/shard.list
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Prepare MUSAN and RIR LMDB augmentation stores if needed."
  if [ ! -d ${musan_lmdb} ]; then
    ${python_cmd} tools/make_lmdb.py ${musan_wav_scp} ${musan_lmdb}
  fi
  if [ ! -d ${rir_lmdb} ]; then
    ${python_cmd} tools/make_lmdb.py ${rir_wav_scp} ${rir_lmdb}
  fi
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Start baseline ResNet34 training."
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  global_batch=$((num_gpus * batch_size))
  if [ "${global_batch}" -ne 384 ]; then
    echo "This baseline requires effective global batch size 384; got ${global_batch} (${num_gpus} GPUs x batch_size ${batch_size})." >&2
    exit 1
  fi
  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  ${torchrun_cmd} --nnodes=$num_nodes --nproc_per_node=$num_gpus \
           --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
    wespeaker/bin/train.py --config $config \
      --exp_dir ${exp_dir} \
      --gpus $gpus \
      --num_avg ${num_avg} \
      --data_type "${data_type}" \
      --train_data ${train_dir}/${data_type}.list \
      --train_label ${train_dir}/utt2spk \
      --key_filter_file ${train_dir}/key.list \
      --reverb_data ${rir_lmdb} \
      --noise_data ${musan_lmdb} \
      ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Average model and extract VoxCeleb1 embeddings."
  avg_model=$exp_dir/models/avg_model.pt
  ${python_cmd} wespeaker/bin/average_model.py \
    --dst_model $avg_model \
    --src_path $exp_dir/models \
    --num ${num_avg}

  vox1_wavs_num=$(wc -l ${vox1_dir}/wav.scp | awk '{print $1}')
  bash tools/extract_embedding.sh \
    --exp_dir $exp_dir \
    --model_path $avg_model \
    --data_type $data_type \
    --data_list ${vox1_dir}/${data_type}.list \
    --wavs_num ${vox1_wavs_num} \
    --store_dir vox1 \
    --batch_size 1 \
    --num_workers 1 \
    --nj ${extract_nj} \
    --gpus "$gpus"
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Extract baseline training embeddings for mean subtraction."
  train_wavs_num=$(wc -l ${train_dir}/wav.scp | awk '{print $1}')
  bash tools/extract_embedding.sh \
    --exp_dir $exp_dir \
    --model_path $exp_dir/models/avg_model.pt \
    --data_type $data_type \
    --data_list ${train_dir}/${data_type}.list \
    --wavs_num ${train_wavs_num} \
    --store_dir vox2_train \
    --batch_size 1 \
    --num_workers 1 \
    --nj ${extract_nj} \
    --gpus "$gpus"
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Score baseline with cosine scoring only."
  rm -f ${exp_dir}/scores/baseline_cos_result
  bash local/score_baseline.sh \
    --stage 1 --stop-stage 2 \
    --exp_dir $exp_dir \
    --trials "$trials" \
    --trials_dir ${work_data}/trials \
    --cal_mean ${use_mean_subtraction} \
    --ptarget ${ptarget} \
    --cfa ${cfa} \
    --cmiss ${cmiss} \
    --python_cmd ${python_cmd}
fi
