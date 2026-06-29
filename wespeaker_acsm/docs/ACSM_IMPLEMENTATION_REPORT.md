# ACSM-Net Implementation Report

## Scope

ACSM-Net is implemented as `ResNet34_ACSM`, a structural variant of the
official WeSpeaker ResNet34. It keeps the original ResNet stem, residual
layers, pooling, and segment layers intact, then adds age observation,
age-conditioned FiLM, and ordered age-to-canonical embedding transformation.

The existing `ResNet34`, extraction protocol, scoring code, and trial-list
logic remain compatible.

## Inspection Summary

The implementation was based on these existing code paths:

- `wespeaker/models/resnet.py`: ResNet34 construction, `_get_frame_level_feat`,
  pooling, `seg_1`, and `seg_2`.
- `wespeaker/models/speaker_model.py`: model-name factory dispatch.
- `wespeaker/models/acsm_modules.py`: ACSM-specific ordinal age observer,
  canonicalizer, and loss helpers.
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
- `wespeaker/bin/train.py`: ACSM config normalization, age-label validation,
  JIT skip for ACSM.
- `wespeaker/utils/executor.py`: ACSM extra-loss branch and ACSM diagnostics.
- `wespeaker/bin/extract.py`: ACSM config normalization before model creation.
- `examples/voxceleb/v2/conf/resnet34_acsm.yaml`: example config.
- `examples/voxceleb/v2/conf/resnet34_acsm_main.yaml`: v2 main config.
- `examples/voxceleb/v2/conf/resnet34_acsm_main_v3.yaml`: current v3 config.
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

## ResNet34_ACSM Construction Closure

The ACSM model entry is `ResNet34_ACSM` in `wespeaker/models/resnet.py`:

```python
def ResNet34_ACSM(feat_dim,
                  embed_dim,
                  pooling_func='TSTP',
                  two_emb_layer=False,
                  acsm_args=None):
    return ResNetACSM(BasicBlock, [3, 4, 6, 3],
                      feat_dim=feat_dim,
                      embed_dim=embed_dim,
                      pooling_func=pooling_func,
                      two_emb_layer=two_emb_layer,
                      acsm_args=acsm_args)
```

`get_speaker_model("ResNet34_ACSM")` resolves through the normal ResNet
factory branch in `wespeaker/models/speaker_model.py`, so a config containing
`model: ResNet34_ACSM` is built by the same training path as the official
WeSpeaker ResNet family. `resnet34_acsm_main.yaml` has been smoke-tested by
loading the YAML, normalizing `acsm_args` with `get_acsm_config`, constructing
the returned model class, and forwarding fake `[B, T, F] = [2, 200, 80]`
features.

ACSM is a structural ResNet34 variant: it subclasses the official `ResNet`,
keeps the shared stem/layers/pooling/segment layers, and inserts ACSM-specific
modules in the ResNet forward path.

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

During extraction, `wespeaker/bin/extract.py` calls `outputs = model(features)`
without passing `age_group`. For dict outputs it selects
`outputs["embedding"]`; therefore ACSM extraction uses the predicted posterior
from `Stage2AgeObserver` and does not require true test age or trial age
information.

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

## Legacy Module Policy

Legacy removed modules and configs have been removed from this branch. Current
experiments are baseline vs ACSM v1/v2/v3.

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
- Canonical trajectory claims require effectiveness diagnostics in addition to
  training losses.

## Recommended Small-Scale Experiment Order

1. Run unit tests and fake-batch backward smoke tests.
2. Run a short ACSM training smoke test with `lambda_age=0` and no age labels.
3. Run a short supervised smoke test with a small age-label file.
4. Compare baseline-compatible extraction and scoring scripts.
5. Only then run small EER experiments.

Recommended config order:

1. `resnet34_acsm.yaml`
2. `resnet34_acsm_main.yaml`
3. `resnet34_acsm_main_v3.yaml`

## Verification Status

Current verification covers:

- `get_speaker_model("ResNet34_ACSM")` returns the ACSM factory and constructs
  a real ACSM model from `resnet34_acsm_main.yaml`.
- `AgeFiLM2d` identity initialization and disabled bypass behavior.
- `Stage2AgeObserver` posterior shape, normalization, non-negativity, finite
  ordinal loss, and all-ignore age loss.
- `OrderedAgeCanonicalizer` reference residual, gate range, normalized output,
  `canonical_scale=0`, and smooth loss.
- `PathConsistencyLoss` same-speaker different-age pairs, no-pair zero loss,
  valid pair count, and ignore-age exclusion.
- `ResNet34_ACSM` forward without age labels.
- ACSM speaker-loss plus extra-loss backward pass with gradients on age
  observer, FiLM, canonicalizer gate, and transition parameters.
- Extraction-style dict-output handling through `outputs["embedding"]` without
  passing true age labels.
- Baseline `ResNet34` still returns its original tuple output.

Commands:

```bash
cd wespeaker_acsm
python -m pytest tests/test_acsm.py tests/test_acsm_effectiveness_diagnostics.py -q
```

In the active `/xmudata/pzj/envs/casv1` environment this command currently
fails before test execution because `pytest` is not installed:
`No module named pytest`.

The same test files are `unittest` compatible and were run with:

```bash
cd wespeaker_acsm
/xmudata/pzj/envs/casv1/bin/python -m unittest tests.test_acsm
```

The fake-batch smoke used features shaped `[4, 200, 80]`, dummy speaker
labels, and valid age groups; it verified finite losses, finite embeddings,
normalized `age_posterior`, gate values within `[0, gate_max]`, finite
gradients, and extraction-style forward without age labels. The extraction
smoke used `examples/voxceleb/v2/data/baseline/extract_smoke/raw.list` and
wrote finite 256-dimensional normalized embeddings.

No real EER experiment or full-scale training has been run as part of this
implementation. Passing these implementation tests does not establish
performance improvement.
