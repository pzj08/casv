# ACSM Experiment Plan

## Main Matrix

| System | Config | Seed | Checkpoint | Vox-O EER/minDCF | Vox-E | Vox-H | CA5 | CA10 | CA15 | CA20 | Params | Latency | Diagnostics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet34 baseline | `baseline_resnet34.yaml` | 3407 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | N/A |
| AORC/OATC | existing AORC config | 3407 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | AORC diag |
| ACSM | `resnet34_acsm.yaml` | 3407 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | gate/residual |
| ACSM-safe | `resnet34_acsm_safe.yaml` | 3407 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | gate/residual |
| ACSM-path | `resnet34_acsm_path.yaml` | 3407 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | path pairs |
| ACSM-aggressive | `resnet34_acsm_aggressive.yaml` | 3407 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | gate/residual |

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

## Ablations

- `resnet34_acsm_no_film.yaml`: remove AgeFiLM.
- `resnet34_acsm_no_canonicalizer.yaml`: remove canonical scoring.
- `resnet34_acsm_no_age_loss.yaml`: remove supervised age loss.
- `resnet34_acsm_no_consistency.yaml`: remove ordinary-SV consistency guard.
- `resnet34_acsm_path.yaml`: compare `lambda_path=0` against weak path
  consistency.
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

1. Run `resnet34_acsm_safe.yaml` for a short smoke run.
2. Run `resnet34_acsm.yaml` as the main ACSM candidate.
3. Run `resnet34_acsm_path.yaml` only after pair coverage is acceptable.
4. Run ablations: no FiLM, no canonicalizer, no age loss, no consistency.
5. Run aggressive only after ordinary SV degradation is understood.
