# ACSM Experiment Plan

## Main Matrix

| System | Config | Seed | Checkpoint | Vox-O EER/minDCF | Vox-E | Vox-H | CA5 | CA10 | CA15 | CA20 | Params | Latency | Diagnostics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet34 baseline | `baseline_resnet34.yaml` | 3407 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | N/A |
| ACSM v1 | `resnet34_acsm.yaml` | 3407 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | effectiveness diagnostics |
| ACSM v2 | `resnet34_acsm_main.yaml` | 3407 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | effectiveness diagnostics |
| ACSM v3 | `resnet34_acsm_main_v3.yaml` | 3407 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | effectiveness diagnostics |

## Test Sets

Use the Vox-CA/MIM evaluation protocol:

- Ordinary SV: `vox1_O_cleaned.kaldi`, `vox1_E_cleaned.kaldi`,
  `vox1_H_cleaned.kaldi`.
- Cross-age: `only_ca5/10/15/20.kaldi` and `vox_ca5/10/15/20.kaldi`.

## Metrics

- EER.
- minDCF.
- actDCF if supported by the current scoring tooling.
- Parameter count.
- Forward latency.
- RTF if measured by a separate extraction benchmark.
- ACSM diagnostics: gate mean/std, residual norm, raw/canonical cosine,
  path-valid pair count.

## Diagnostics

- Use `tools/diagnose_acsm_trajectory.py --effectiveness-report` for
  raw-vs-canonical change, same-speaker cross-age distance, different-speaker
  preservation, near-identity risk, and collapse risk.
- Predicted age vs oracle age can only be reported as diagnostic, never as the
  main fair result.

## Multi-Seed

Run at least three seeds before making a performance claim:

- seed 3407
- seed 3408
- seed 3409

## Statistical Significance

Use paired bootstrap or trial-level bootstrap before claiming small gains. If
this is not implemented, mark significance analysis as future required.

## Result Naming

Do not overwrite old results. Each experiment directory must keep:

- `config.yaml`
- `experiment_manifest.yaml`
- git commit hash
- seed
- checkpoint path
- score files
- metric files
- diagnostics JSON

## Recommended Order

1. Run `baseline_resnet34.yaml`.
2. Run `resnet34_acsm.yaml` only as the retained v1 reference.
3. Run `resnet34_acsm_main.yaml` as the v2 comparison point.
4. Run `resnet34_acsm_main_v3.yaml` as the current candidate.
5. Run effectiveness diagnostics before making any mechanistic claim.
