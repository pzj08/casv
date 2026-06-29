# ACSM Handoff - 2026-06-28

Project root:

`/xmudata/pzj/casv`

Main working directory:

`/xmudata/pzj/casv/wespeaker_aorc`

Current stage: ACSM-Net has passed implementation-level repair and has entered engineering validation plus first real training runs. There are no official EER/minDCF results yet. Do not claim ACSM improves performance.

## 1. Main Changes In This Round

### 1.1 Consistency Loss Fixed

The ACSM consistency loss was changed from unnormalized raw L2 over embedding dimensions to normalized cosine consistency.

Main file:

`wespeaker_aorc/wespeaker/models/resnet.py`

Current intended behavior:

- normalize canonical embedding `outputs["embedding"]`
- normalize detached raw embedding `outputs["raw_embedding"].detach()`
- compute `1 - cosine_similarity`
- keep `lambda_consistency=0.03`

Reason:

The old raw L2 sum produced very large values. With 256-dimensional embeddings, it dominated total loss after weighting and forced canonical embedding close to raw embedding. The corrected cosine loss keeps consistency as a weak ordinary-SV preservation constraint.

### 1.2 ACSM Main Config Updated

Main config:

`wespeaker_aorc/examples/voxceleb/v2/conf/resnet34_acsm_main.yaml`

Important current parameters:

```yaml
enable_amp: True

acsm_args:
  canonicalizer:
    canonical_scale: 0.15
    gate_max: 0.40
    gate_init_bias: -1.5
    transition_init_std: 0.01

  film:
    film_scale: 0.05

  losses:
    lambda_age: 0.10
    lambda_consistency: 0.03
    lambda_smooth: 1.0e-4
    lambda_path: 0.01
    ramp_epoch: 3

  consistency:
    type: cosine
```

Do not revert to raw L2 consistency as main ACSM. If kept, raw L2 should only be a legacy/control ablation.

### 1.3 Training Logs Extended

Main file:

`wespeaker_aorc/wespeaker/utils/executor.py`

ACSM logs include or are expected to include:

- `loss_age`
- `loss_consistency`
- `weighted_consistency`
- `loss_smooth`
- `loss_path`
- `loss_acsm_total`
- `path_valid_pair_count`
- `path_nonzero_batch_ratio`
- `gate_mean`
- `residual_norm_mean`
- `raw_can_cosine_mean`
- `l2_raw_can_mean`

Current logging precision for very small values may be insufficient. For detailed diagnosis, change `loss_consistency`, `weighted_consistency`, and `l2_raw_can_mean` to scientific notation.

## 2. Shard Data Migration

Original shard path:

`/data/vox2_dev/shards`

New shard path:

`/work1/pzj/vox2_dev/shards`

Full copy completed:

- 1093 shards
- about 256G
- `/work1` is non-rotational and tested faster than `/data`

Default shard list was overwritten:

`wespeaker_aorc/examples/voxceleb/v2/data/baseline/vox2_train_voxca/shard.list`

Current status:

- `shard.list` has 1093 lines
- all entries point to `/work1/pzj/vox2_dev/shards/*.tar`
- no `/data/` entries remain

The previous helper list still exists:

`wespeaker_aorc/examples/voxceleb/v2/data/baseline/vox2_train_voxca/shard_work1_full.list`

Future default training that uses `shard.list` will use `/work1` without passing a custom `--train_data`.

## 3. Current ACSM Training Run

Current run/session:

`acsm_main_work1_gpu4_7`

Experiment directory:

`/xmudata/pzj/casv/wespeaker_aorc/exp/ACSM-ResNet34-main-cos-work1-gpu4-7-20260628-v2`

Launch log:

`/xmudata/pzj/casv/wespeaker_aorc/exp_launch_logs/acsm_main_work1_gpu4_7_20260628_v2.log`

Train log:

`/xmudata/pzj/casv/wespeaker_aorc/exp/ACSM-ResNet34-main-cos-work1-gpu4-7-20260628-v2/train.log`

Physical GPUs:

`4,5,6,7`

The active run was launched with an explicit train list pointing to `/work1`. The later default `shard.list` overwrite does not change the already running process, but makes future runs use `/work1` by default.

Observed training behavior so far:

- `loss_spk` decreases normally
- speaker accuracy increases normally
- `loss_age` is finite and decreases
- `loss_path` is finite
- `path_nonzero_batch_ratio` is roughly `0.33-0.35`
- `gate_mean` rises mildly
- no NaN/Inf observed
- speed roughly `5-6 batch/s` after warmup

Observed ACSM diagnostic concern:

- `raw_can_cosine_mean` still displays `1.000000`
- `l2_raw_can_mean` is around `1e-6`
- `weighted_consistency` displays `0.000000`

Because logs use six decimal places, the exact tiny values are not recoverable from old logs. The current state indicates stable optimization, but canonicalizer may still be close to identity. Do not claim trajectory learning yet.

## 4. Important Interpretation

Normalized cosine consistency fixed the major scale bug:

- old raw L2 consistency could dominate total loss
- new cosine consistency no longer appears to dominate speaker loss

However, ACSM effectiveness is not established:

- no EER/minDCF result yet
- no ordinary SV degradation analysis yet
- no path000/path001/path002 comparison yet
- no trajectory diagnosis result yet

Allowed wording:

- ACSM code path is trainable
- ACSM consistency scale issue has been fixed
- ACSM main training is currently stable
- ACSM claims are ready to be tested

Forbidden wording:

- ACSM improves EER
- ACSM outperforms baseline
- ACSM learns true age trajectory
- ACSM does not hurt ordinary SV
- ACSM is proven effective

## 5. Useful Commands

Check tmux:

```bash
tmux ls
```

Check training log:

```bash
tail -80 /xmudata/pzj/casv/wespeaker_aorc/exp/ACSM-ResNet34-main-cos-work1-gpu4-7-20260628-v2/train.log
```

Check launch log:

```bash
tail -40 /xmudata/pzj/casv/wespeaker_aorc/exp_launch_logs/acsm_main_work1_gpu4_7_20260628_v2.log
```

Check GPU:

```bash
nvidia-smi
```

Check checkpoints:

```bash
ls -lh /xmudata/pzj/casv/wespeaker_aorc/exp/ACSM-ResNet34-main-cos-work1-gpu4-7-20260628-v2/models
```

Check default shard list:

```bash
head -5 /xmudata/pzj/casv/wespeaker_aorc/examples/voxceleb/v2/data/baseline/vox2_train_voxca/shard.list
wc -l /xmudata/pzj/casv/wespeaker_aorc/examples/voxceleb/v2/data/baseline/vox2_train_voxca/shard.list
rg -n '^/data/' /xmudata/pzj/casv/wespeaker_aorc/examples/voxceleb/v2/data/baseline/vox2_train_voxca/shard.list
```

Stop current training if explicitly requested:

```bash
tmux kill-session -t acsm_main_work1_gpu4_7
```

## 6. Recommended Next Steps

1. Continue monitoring current ACSM-main training for several epochs.
2. Track:
   - `loss_spk`
   - `Acc`
   - `loss_age`
   - `loss_path`
   - `path_nonzero_batch_ratio`
   - `gate_mean`
   - `residual_norm_mean`
   - `raw_can_cosine_mean`
   - `l2_raw_can_mean`
   - `weighted_consistency`
3. Improve log precision for tiny diagnostics if further ACSM identity-collapse analysis is needed.
4. Run official evaluation only after a proper checkpoint is available.
5. Compare with:
   - ResNet34 baseline
   - ACSM-safe
   - ACSM-main
   - ACSM-path000 (`lambda_path=0`)
   - ACSM-path002 (`lambda_path=0.02`)
   - no_film
   - no_canonicalizer
   - no_consistency
   - no_age_loss
   - AORC
   - ParamMatch if resources allow

Each experiment should save:

- config
- commit hash
- seed
- checkpoint
- score file
- EER/minDCF
- gate/residual diagnostics
- fair evaluation audit
- data leakage audit

## 7. Final Status

Engineering status:

`ACSM training path is currently runnable and stable at the log level observed so far.`

Research status:

`No performance conclusion yet. Formal EER/minDCF and ablation results are still required.`
