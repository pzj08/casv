# Baseline Manifest

This tree is the clean baseline. The source code under `wespeaker/`, `tools/`,
`runtime/`, `docs/`, and the original recipes is the official WeSpeaker clone.
Baseline alignment is limited to recipe-level files under
`examples/voxceleb/v2`.

## Added Recipe Files

- `examples/voxceleb/v2/conf/baseline_resnet34.yaml`
- `examples/voxceleb/v2/local/prepare_baseline_data.py`
- `examples/voxceleb/v2/local/score_baseline.sh`
- `examples/voxceleb/v2/run_baseline_resnet34.sh`

## Baseline Contract

- Model: `ResNet34`
- Features: 80-dim fbank, 25 ms window, 10 ms frame shift
- Augmentation: MUSAN, RIR, speed perturb, 200-frame chunks
- Optimizer: SGD, momentum `0.9`, Nesterov, weight decay `1e-4`
- Schedule: warmup 6 epochs, exponential decrease to effective `5e-5`
- Classification: ArcFace, scale `48`, fixed margin `0.2`
- Batch: 4 GPUs, per-GPU batch size `96`, global batch size `384`
- Scoring: cosine only, no PLDA, no score normalization, no calibration,
  no score fusion, no large-margin fine-tuning

Official WeSpeaker multiplies scheduler LR by
`world_size * per_gpu_batch_size / 64`. With the required `4 x 96` setup,
that multiplier is `6`, so the config stores `0.1 / 6` and `5e-5 / 6`
to make the effective training LR exactly `0.1 -> 5e-5`.

## Acceptance Checks

Run from this tree root:

```bash
rg -n "age_group|acsm_args|ResNet34_ACSM" examples/voxceleb/v2/conf/baseline_resnet34.yaml wespeaker
```

The clean baseline recipe must not enable ACSM-specific model paths.

Run from `examples/voxceleb/v2`:

```bash
bash run_baseline_resnet34.sh --stage 0 --stop-stage 6
```

Stage `4` checks that the extracted VoxCeleb1 embedding count matches
`data/baseline/vox1/wav.scp`. Stage `6` writes cosine-only metrics to
`${exp_dir}/scores/baseline_cos_result`.
