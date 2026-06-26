# ACSM-Net Implementation Report

## Scope

ACSM-Net is implemented as `ResNet34_ACSM`, a structural variant of the
official WeSpeaker ResNet34. It keeps the original ResNet stem, residual
layers, pooling, and segment layers intact, then adds age observation,
age-conditioned FiLM, and ordered age-to-canonical embedding transformation.

This is not an AORC wrapper and does not use AORC as the architectural base.
The existing `ResNet34`, AORC wrapper, extraction protocol, scoring code, and
trial-list logic remain compatible.

## Inspection Summary

The implementation was based on these existing code paths:

- `wespeaker/models/resnet.py`: ResNet34 construction, `_get_frame_level_feat`,
  pooling, `seg_1`, and `seg_2`.
- `wespeaker/models/speaker_model.py`: model-name factory dispatch.
- `wespeaker/models/aorc_modules.py`: AORC wrapper patterns and reusable
  ordinal age ideas only.
- `wespeaker/losses/aorc_losses.py`: reused `OrdinalAgeLoss`.
- `wespeaker/bin/train.py`: age label loading, dataset construction, model
  init/checkpoint flow.
- `wespeaker/utils/executor.py`: dict output handling, speaker loss, extra
  loss logging.
- `wespeaker/bin/extract.py`: dict output extraction via `outputs["embedding"]`.
- `wespeaker/bin/score.py`: cosine scoring; no ACSM changes required.

## Modified Files

- `wespeaker/models/acsm_modules.py`: ACSM modules and config helpers.
- `wespeaker/models/resnet.py`: `ResNetACSM`, `ResNet34_ACSM`,
  `ACSM_ResNet34`.
- `wespeaker/models/speaker_model.py`: model factory support for
  `ACSM_ResNet34`.
- `wespeaker/bin/train.py`: ACSM config normalization, AORC/ACSM mutual
  exclusion, age-label validation, JIT skip for ACSM.
- `wespeaker/utils/executor.py`: ACSM extra-loss branch and ACSM diagnostics.
- `wespeaker/bin/extract.py`: ACSM config normalization before model creation.
- `examples/voxceleb/v2/conf/resnet34_acsm.yaml`: example config.
- `examples/voxceleb/v2/conf/resnet34_acsm_*.yaml`: safe, aggressive, path,
  and ablation configs.
- `tests/test_acsm.py`: ACSM unit and smoke tests.
- `tools/diagnose_acsm.py`: canonicalization activity diagnostics.
- `tools/audit_fair_eval.py`: fair evaluation audit.
- `tools/audit_data_leakage.py`: train/eval/trial leakage audit.
- `tools/diagnose_age_pair_coverage.py`: batch path-pair coverage estimator.
- `tools/profile_model.py`: parameter and latency profiling.

## Modules

- `AgeFiLM2d`: applies small identity-initialized age FiLM to ResNet feature
  maps.
- `Stage2AgeObserver`: predicts CORAL-style ordered age posterior from layer2
  features.
- `OrderedAgeCanonicalizer`: learns adjacent age transitions and maps observed
  embeddings to a reference age group using predicted posterior, uncertainty,
  and a scalar gate.
- `PathConsistencyLoss`: optional same-speaker different-age pairwise cosine
  consistency, disabled by default.

## Losses

`compute_acsm_losses(outputs, speakers, age_groups, epoch)` returns:

- `loss_age`
- `loss_consistency`
- `loss_smooth`
- `loss_path`
- `loss_acsm_total`
- `gate_mean`
- `gate_std`
- `uncertainty_mean`
- `residual_norm`

The executor trains with:

`loss = speaker_loss(outputs["embedding"]) + loss_acsm_total`

The ramp schedule uses `(epoch + 1) / ramp_epoch`, capped at 1.

## Age Labels

Training can use age labels for ordinal age supervision. ACSM requires
`age_label_file` only when `lambda_age > 0` or `lambda_path > 0`. Extraction
does not require or read test age labels.

Following `/xmudata/pzj/MIM.pdf`, the intended training set is the Vox-CA train
set built from VoxCeleb2, not an arbitrary full VoxCeleb2-dev list. In this
workspace that corresponds to
`examples/voxceleb/v2/data/baseline/vox2_train_voxca`, with 5990 speakers and
1085425 utterances. The age values come from
`/xmudata/pzj/vox-ca/vox2dev/segment2age.npy` and are converted with bins
`[21, 31, 41, 51, 61, 71]`, matching the paper's groups: 0-20, 21-30, 31-40,
41-50, 51-60, 61-70, 71-80.

## Fair Evaluation

Extraction uses model-predicted `age_posterior` and writes
`outputs["embedding"]`, the canonical embedding. No oracle test-age path is
implemented by default. `score.py` is unchanged and still performs cosine
scoring over extracted embeddings.

Evaluation should follow the MIM/Vox-CA protocol. The evaluation embeddings are
extracted from `examples/voxceleb/v2/data/baseline/vox1`, and scoring uses the
11 trial files in `examples/voxceleb/v2/data/baseline/trials`:

- VoxCeleb official trials: `vox1_O_cleaned.kaldi`, `vox1_E_cleaned.kaldi`,
  `vox1_H_cleaned.kaldi`.
- Cross-age trials: `only_ca5.kaldi`, `only_ca10.kaldi`, `only_ca15.kaldi`,
  `only_ca20.kaldi`, `vox_ca5.kaldi`, `vox_ca10.kaldi`, `vox_ca15.kaldi`,
  `vox_ca20.kaldi`.

These local trial files match the MIM paper counts. The Vox-CA train set
`vox2_train_voxca` may be extracted for mean subtraction only; it is not an
evaluation trial set.

Run the fair-evaluation audit before reporting metrics:

```bash
python tools/audit_fair_eval.py \
  --config examples/voxceleb/v2/conf/resnet34_acsm.yaml \
  --score-py wespeaker/bin/score.py \
  --trial-list examples/voxceleb/v2/data/baseline/trials/vox1_O_cleaned.kaldi
```

## AORC Relationship

AORC remains a separate wrapper and baseline/ablation path. ACSM is not
implemented by wrapping `AORCWrapper`, and ACSM/AORC are rejected when enabled
together.

## Known Risks

- Age label quality directly affects the age observer.
- Too-large canonical residuals may degrade ordinary ASV behavior.
- Gate saturation can suppress or over-apply canonicalization.
- Inaccurate age posterior can move embeddings in the wrong direction.
- Optional loss branches should be monitored under DDP for unused parameters.
- Baseline checkpoint initialization is partial: shared ResNet keys load,
  ACSM-specific keys are newly initialized.
- If `gate_mean` stays near zero and raw/canonical cosine stays near 1,
  canonicalization may be near identity.
- If `lambda_path=0`, canonical trajectory claims require additional ablation;
  use `resnet34_acsm_path.yaml` for weak path consistency.

## Recommended Small-Scale Experiment Order

1. Run unit tests and fake-batch backward smoke tests.
2. Run a short ACSM training smoke test with `lambda_age=0` and no age labels.
3. Run a short supervised smoke test with a small age-label file.
4. Compare baseline-compatible extraction and scoring scripts.
5. Only then run small EER experiments.

Recommended config order:

1. `resnet34_acsm_safe.yaml`
2. `resnet34_acsm.yaml`
3. `resnet34_acsm_path.yaml`
4. `resnet34_acsm_no_film.yaml`
5. `resnet34_acsm_no_canonicalizer.yaml`
6. `resnet34_acsm_no_age_loss.yaml`
7. `resnet34_acsm_no_consistency.yaml`
8. `resnet34_acsm_aggressive.yaml`

## Verification Status

This implementation has passed unit tests, a fake-batch forward/backward smoke
test, and an extraction smoke test with a temporary ACSM checkpoint. The
fake-batch smoke used features shaped `[4, 200, 80]`, dummy speaker labels, and
valid age groups; it verified finite losses, finite embeddings, normalized
`age_posterior`, gate values within `[0, gate_max]`, finite gradients, and
extraction-style forward without age labels. The extraction smoke used
`examples/voxceleb/v2/data/baseline/extract_smoke/raw.list` and wrote 8 finite
256-dimensional normalized embeddings.

No real EER experiment or full-scale training has been run as part of this
implementation.
