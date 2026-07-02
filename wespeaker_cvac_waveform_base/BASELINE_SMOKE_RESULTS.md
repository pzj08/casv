# Baseline Smoke Results

Date: 2026-06-19

## Scope

This is a connectivity smoke run, not a formal result. It used one training
iteration over `384` samples and then exercised full VoxCeleb1 extraction and
all configured VoxCeleb1/Vox-CA cosine scoring paths.

## Artifacts

- Experiment: `examples/voxceleb/v2/exp/smoke_baseline_20260619_210900`
- Model: `models/avg_model.pt`
- VoxCeleb1 embeddings: `embeddings/vox1/xvector.scp`
- Score summary: `scores/baseline_cos_result`

## Checks

- 4GPU train completed with per-GPU batch size `96`.
- VoxCeleb1 extraction completed.
- Embedding count matched wav count: `153516 / 153516`.
- Scoring completed for Vox-O/E/H and Vox-CA trial files.

## Smoke Metrics

These numbers only confirm that scoring ran end to end after a tiny training
run.

| Trial | EER | minDCF |
| --- | ---: | ---: |
| vox1_O_cleaned | 40.752 | 0.993 |
| vox1_E_cleaned | 39.905 | 0.993 |
| vox1_H_cleaned | 41.784 | 0.996 |
| only_ca5 | 45.974 | 1.000 |
| only_ca10 | 48.523 | 1.000 |
| only_ca15 | 49.683 | 1.000 |
| only_ca20 | 49.488 | 0.999 |
| vox_ca5 | 48.226 | 1.000 |
| vox_ca10 | 50.152 | 1.000 |
| vox_ca15 | 51.340 | 1.000 |
| vox_ca20 | 52.086 | 1.000 |
