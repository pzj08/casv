# AORC Smoke Results

Date: 2026-06-19

## Scope

This is a connectivity smoke run for the derived AORC tree. It used the same
baseline smoke data view and one training iteration over `384` samples.

## AORC Off

- Experiment: `examples/voxceleb/v2/exp/smoke_aorc_off_20260619_210900`
- 4GPU train completed with per-GPU batch size `96`.
- `models/model_1.pt` and `models/final_model.pt` were written.
- Log check: no `AORC loss_`, `AORC enabled`, `age label num`, or
  `AORCWrapper` entries appeared.

## AORC On

- Experiment: `examples/voxceleb/v2/exp/smoke_aorc_on_20260619_210900`
- 4GPU train completed with per-GPU batch size `96`.
- Age labels loaded: `age label num: 20000, num_age_groups: 7`.
- AORC losses were non-zero, for example:
  `loss_oam=0.799278`, `loss_caa=4.600470`,
  `loss_smooth=0.052400`.
- Small extraction smoke completed with `8 / 8` xvectors written to
  `embeddings/extract_smoke/xvector.scp`.
