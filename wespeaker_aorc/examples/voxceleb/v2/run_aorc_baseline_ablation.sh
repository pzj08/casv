#!/bin/bash

# Copyright 2026
# Licensed under the Apache License, Version 2.0.

. ./path.sh || exit 1

stage=0
stop_stage=3

HOST_NODE_ADDR_BASE=29600
num_nodes=1
job_id_base=7600
gpus="[0,1,2,3]"

data=/data
work_data=data/baseline
data_type=shard
train_dir=${work_data}/vox2_train_voxca
vox1_dir=${work_data}/vox1
musan_lmdb=/data/musan/lmdb
rir_lmdb=/data/rirs/lmdb

num_epochs=3
num_avg=3
sample_num_per_epoch=65536
batch_size=96
num_workers=16
prefetch_factor=8
log_batch_interval=20
extract_nj=4
python_cmd=python
torchrun_cmd=torchrun

run_id=$(date +%Y%m%d_%H%M%S)
run_root=exp/aorc_baseline_ablation_${run_id}
variants_filter=

trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi only_ca5.kaldi only_ca10.kaldi only_ca15.kaldi only_ca20.kaldi vox_ca5.kaldi vox_ca10.kaldi vox_ca15.kaldi vox_ca20.kaldi"
use_mean_subtraction=true
ptarget=0.01
cfa=1
cmiss=1

. tools/parse_options.sh || exit 1
set -e
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

variants=(
  "aorc_off:conf/baseline_resnet34_aorc_off.yaml"
  "oam:conf/baseline_resnet34_oam.yaml"
  "age_ce:conf/baseline_resnet34_age_ce.yaml"
  "oam_orc:conf/baseline_resnet34_oam_orc.yaml"
  "oam_caa:conf/baseline_resnet34_oam_caa.yaml"
  "aorc_full_wo_oam:conf/baseline_resnet34_aorc_full_wo_oam.yaml"
  "aorc_full:conf/baseline_resnet34_aorc_full.yaml"
)

mkdir -p "${run_root}/configs"
timing_tsv="${run_root}/timing.tsv"
summary_tsv="${run_root}/metrics.tsv"
[ -f "${timing_tsv}" ] || echo -e "variant\tphase\tseconds" >"${timing_tsv}"
[ -f "${summary_tsv}" ] || echo -e "variant\ttest_set\teer\tmindcf" >"${summary_tsv}"

should_run_variant() {
  local wanted
  [ -z "${variants_filter}" ] && return 0
  for wanted in ${variants_filter}; do
    [ "${wanted}" = "$1" ] && return 0
  done
  return 1
}

make_config() {
  local template=$1
  local out_config=$2
  local exp_dir=$3
  ${python_cmd} - "$template" "$out_config" "$exp_dir" "$gpus" "$num_epochs" "$num_avg" "$sample_num_per_epoch" "$batch_size" "$num_workers" "$prefetch_factor" "$log_batch_interval" "${train_dir}/key.list" <<'PY'
import sys
from pathlib import Path

import yaml

template, out_config, exp_dir, gpus = sys.argv[1:5]
num_epochs, num_avg = map(int, sys.argv[5:7])
sample_num_per_epoch, batch_size, num_workers = map(int, sys.argv[7:10])
prefetch_factor, log_batch_interval = map(int, sys.argv[10:12])
key_filter_file = sys.argv[12]

with open(template, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

gpu_list = yaml.safe_load(gpus)
if not isinstance(gpu_list, list):
    raise ValueError("gpus must be a list, got {!r}".format(gpus))

cfg["exp_dir"] = exp_dir
cfg["gpus"] = gpu_list
cfg["num_epochs"] = num_epochs
cfg["num_avg"] = num_avg
cfg["log_batch_interval"] = log_batch_interval
cfg["key_filter_file"] = key_filter_file
cfg.setdefault("dataset_args", {})["sample_num_per_epoch"] = sample_num_per_epoch
cfg.setdefault("dataloader_args", {})["batch_size"] = batch_size
cfg["dataloader_args"]["num_workers"] = num_workers
cfg["dataloader_args"]["prefetch_factor"] = prefetch_factor

Path(out_config).parent.mkdir(parents=True, exist_ok=True)
with open(out_config, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
}

parse_metrics() {
  local variant=$1
  local result_file=$2
  ${python_cmd} - "$variant" "$result_file" "$summary_tsv" <<'PY'
import re
import sys
from pathlib import Path

variant, result_file, summary_tsv = sys.argv[1:4]
rows = []
current = None
eer = None

for line in Path(result_file).read_text(encoding="utf-8", errors="ignore").splitlines():
    m = re.match(r"-+ (.+?) -----", line.strip())
    if m:
        current = Path(m.group(1)).stem
        if current.endswith(".kaldi"):
            current = current[:-6]
        eer = None
        continue
    if current and line.startswith("EER ="):
        eer = float(line.split("=")[1].strip())
        continue
    if current and line.startswith("minDCF") and eer is not None:
        mindcf = float(line.rsplit("=", 1)[1].strip())
        rows.append((variant, current, eer, mindcf))
        current = None

summary_path = Path(summary_tsv)
existing = summary_path.read_text(encoding="utf-8").splitlines()
kept = [line for i, line in enumerate(existing) if i == 0 or not line.startswith(variant + "\t")]
with summary_path.open("w", encoding="utf-8") as f:
    f.write("\n".join(kept).rstrip() + "\n")
    for row in rows:
        f.write("%s\t%s\t%.6f\t%.6f\n" % row)
PY
}

time_phase() {
  local variant=$1
  local phase=$2
  shift 2
  local start end
  start=$(date +%s)
  "$@"
  end=$(date +%s)
  echo -e "${variant}\t${phase}\t$((end - start))" >>"${timing_tsv}"
}

for idx in "${!variants[@]}"; do
  entry=${variants[$idx]}
  variant=${entry%%:*}
  template=${entry#*:}
  exp_dir="${run_root}/${variant}"
  config="${run_root}/configs/${variant}.yaml"
  port=$((HOST_NODE_ADDR_BASE + idx))
  job_id=$((job_id_base + idx))

  if ! should_run_variant "${variant}"; then
    echo "[$variant] skipped by variants_filter=${variants_filter}"
    continue
  fi

  if [ ${stage} -le 1 ]; then
    make_config "$template" "$config" "$exp_dir"
  elif [ ! -f "$config" ]; then
    echo "[$variant] warning: ${config} not found; stage ${stage} does not need to regenerate it."
  fi

  if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "[$variant] train: config=${config} exp_dir=${exp_dir}"
    num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
    global_batch=$((num_gpus * batch_size))
    if [ "${global_batch}" -ne 384 ]; then
      echo "AORC Baseline ablation requires effective global batch size 384; got ${global_batch} (${num_gpus} GPUs x batch_size ${batch_size})." >&2
      exit 1
    fi
    time_phase "$variant" "train" \
      ${torchrun_cmd} --nnodes=$num_nodes --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend=c10d --rdzv_endpoint=localhost:$port \
        wespeaker/bin/train.py --config "$config" \
          --data_type "$data_type" \
          --train_data "${train_dir}/${data_type}.list" \
          --train_label "${train_dir}/utt2spk" \
          --reverb_data "$rir_lmdb" \
          --noise_data "$musan_lmdb"
  fi

  if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "[$variant] average and extract VoxCeleb1/train embeddings"
    avg_model="${exp_dir}/models/avg_model.pt"
    time_phase "$variant" "average" \
      ${python_cmd} wespeaker/bin/average_model.py \
        --dst_model "$avg_model" --src_path "${exp_dir}/models" --num "$num_avg"

    vox1_wavs_num=$(wc -l "${vox1_dir}/wav.scp" | awk '{print $1}')
    time_phase "$variant" "extract_vox1" \
      bash tools/extract_embedding.sh \
        --exp_dir "$exp_dir" \
        --model_path "$avg_model" \
        --data_type "$data_type" \
        --data_list "${vox1_dir}/${data_type}.list" \
        --wavs_num "$vox1_wavs_num" \
        --store_dir vox1 \
        --batch_size 1 \
        --num_workers 1 \
        --nj "$extract_nj" \
        --gpus "$gpus"

    train_wavs_num=$(wc -l "${train_dir}/wav.scp" | awk '{print $1}')
    time_phase "$variant" "extract_vox2_train" \
      bash tools/extract_embedding.sh \
        --exp_dir "$exp_dir" \
        --model_path "$avg_model" \
        --data_type "$data_type" \
        --data_list "${train_dir}/${data_type}.list" \
        --wavs_num "$train_wavs_num" \
        --store_dir vox2_train \
        --batch_size 1 \
        --num_workers 1 \
        --nj "$extract_nj" \
        --gpus "$gpus"
  fi

  if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "[$variant] cosine-only scoring"
    rm -f "${exp_dir}/scores/baseline_cos_result"
    time_phase "$variant" "score" \
      bash local/score_baseline.sh \
        --stage 1 --stop-stage 2 \
        --exp_dir "$exp_dir" \
        --trials "$trials" \
        --trials_dir "${work_data}/trials" \
        --cal_mean "${use_mean_subtraction}" \
        --ptarget "${ptarget}" \
        --cfa "${cfa}" \
        --cmiss "${cmiss}" \
        --python_cmd "${python_cmd}"
    parse_metrics "$variant" "${exp_dir}/scores/baseline_cos_result"
  fi
done

echo "AORC Baseline ablation complete."
echo "Metrics: ${summary_tsv}"
echo "Timing: ${timing_tsv}"
