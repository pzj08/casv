# ACSM Risk Fix Strict Review Report

Date: 2026-06-26

Scope: strict review of the ACSM high-risk fixes in `wespeaker_acsm`. This review checks whether the risk was actually reduced by executable code, tests, diagnostics, and configuration, not only by documentation. No formal training result or EER/minDCF improvement is claimed here.

## Executive Summary

| Area | Status | Review conclusion |
| --- | --- | --- |
| 1. ACSM unit tests | PARTIAL | Coverage exists and passes under `unittest`; requested `pytest` command could not run because `pytest` is not installed in the active environment. |
| 2. Empty-module diagnostics | PARTIAL | Diagnostics exist and report gate/residual/cosine/path stats, but current untrained/smoke checkpoints still show near-identity behavior; effectiveness is detectable, not proven. |
| 3. Canonical trajectory claim | PARTIAL | Weak path loss and config exist with correct pair constraints, but no real path ablation result yet. |
| 4. AgeFiLM ablations | PASS | Required ablation configs exist and pass config/model construction smoke tests. |
| 5. Fair evaluation | PARTIAL | Default extraction/scoring do not use true test age, and audit tooling exists; oracle-age enforcement is mostly audit/config-level because oracle eval is not a normal implemented path. |
| 6. Data leakage audit | PARTIAL | Speaker/utterance/trial checks work; age-label overlap check is exact-key based and can undercount when dataset keys use alternate normalized forms. |
| 7. Batch pair coverage | PASS | Pair coverage diagnostic exists, reports nonzero-batch ratio, and sampler was not changed. |
| 8. Checkpoint loading | PASS | Partial ResNet34-to-ACSM load is implemented with strict/non-strict behavior and tests. |
| 9. Ordinary SV protection | PARTIAL | Consistency loss, logging, and safe/aggressive configs exist; no ordinary SV metric result yet. |
| 10. Model profile | PASS | Parameter and latency profiling works; FLOPs are explicitly reported as unavailable. |
| 11. Experiment plan | PASS | Plan covers main experiments, ablations, metrics, seeds, CA gaps, and result naming. |
| 12. Claim guidelines | PASS | Over-claims are explicitly forbidden and evidence requirements are documented. |
| 13. Reproducibility | PARTIAL | Config, seed, manifest, commit hash, ACSM hparams are recorded; dirty worktree state and DataLoader worker seeding remain weak points. |

## 1. ACSM Unit Tests

Status: PARTIAL

Checks:

| Check | Status | Evidence |
| --- | --- | --- |
| `tests/test_acsm.py` exists | PASS | `wespeaker_acsm/tests/test_acsm.py` exists. |
| Covers `AgeFiLM2d` | PASS | Tests enabled/disabled behavior, shape, finite values, and near-identity zero initialization. |
| Covers `Stage2AgeObserver` | PASS | Tests output keys, posterior normalization/non-negativity, ordinal loss, and ignore-age behavior. |
| Covers `OrderedAgeCanonicalizer` | PASS | Tests shape, gate range, L2 normalization, reference-age residual, smooth loss, and `canonical_scale=0`. |
| Covers `ResNet34_ACSM` forward | PASS | Tests dict output without `age_group`, required keys, shapes, and finite tensors. |
| Covers ACSM loss backward | PASS | Tests age/consistency/smooth/path/total losses, backward pass, and gradients on ACSM parameters. |
| Covers baseline compatibility | PASS | Tests baseline `ResNet34` behavior and dict-output extraction assumption. |
| CPU independent | PASS | Tests use fake tensors and CPU by default. |
| No age label / all-ignore age | PASS | Covered in forward and loss tests. |
| Requested `pytest` actually ran | FAIL | `pytest` is not installed in the active environment. |

Commands and results:

```bash
/xmudata/pzj/envs/casv1/bin/python -m unittest tests.test_acsm
```

Result: PASS, 26 tests.

```bash
/xmudata/pzj/envs/casv1/bin/python -m pytest tests/test_acsm.py tests/test_acsm_effectiveness_diagnostics.py -q
```

Result: FAIL before test execution: `No module named pytest`.

Review conclusion: the test content is meaningful and executable via the standard library test runner, but the exact requested pytest command is not currently runnable in this environment.

## 2. ACSM Empty-Module Diagnostics

Status: PARTIAL

Script: `wespeaker_acsm/tools/diagnose_acsm.py`

Example command:

```bash
/xmudata/pzj/envs/casv1/bin/python tools/diagnose_acsm.py \
  --config examples/voxceleb/v2/conf/resnet34_acsm.yaml \
  --checkpoint /tmp/acsm_risk_smoke/acsm.pt \
  --fake --batch-size 4 --num-frames 80 --device cpu
```

Checks:

| Check | Status | Evidence |
| --- | --- | --- |
| Gate/residual/uncertainty diagnostics | PASS | Reports `gate_mean`, `gate_std`, `gate_min`, `gate_max`, `uncertainty_mean`, residual stats. |
| Reports `raw_can_cosine_mean` | PASS | JSON/text report includes it. |
| Reports `residual_norm_mean` | PASS | JSON/text report includes it. |
| Reports age-group path norm | PASS | Includes `path_norm_by_age_group`. |
| Can judge near-identity canonicalizer | PASS | Reports cosine, L2 distance, residual norm, and gate stats; docs explain interpretation. |
| Does not use true test age to alter output | PASS | Age labels are used only for grouping diagnostics when available; model inference uses predicted posterior. |

Observed diagnostic behavior: fake/untrained smoke checkpoints produced near-identity values such as `raw_can_cosine_mean` near 1 and very small raw/canonical distance. This means the diagnostic can expose the risk, but it does not prove ACSM is active after real training.

## 3. Canonical Trajectory Claim

Status: PARTIAL

Checks:

| Check | Status | Evidence |
| --- | --- | --- |
| Stable `lambda_path=0` retained | PASS | `resnet34_acsm.yaml` keeps `lambda_path: 0.0`. |
| `lambda_path>0` config exists | PASS | `resnet34_acsm_main.yaml` and `resnet34_acsm_main_v3.yaml` set weak path consistency. |
| Path loss uses same speaker + different age | PASS | `PathConsistencyLoss.valid_pair_indices()` filters same speaker, valid age, and different age. |
| No valid pair returns zero | PASS | Loss returns a same-device zero tensor when no pair exists. |
| Valid pair count logged | PASS | `path_valid_pair_count` is returned by ACSM loss and logged by the executor. |
| Docs limit trajectory claims | PASS | Docs state that path/cross-age evidence is required before claiming learned age trajectories. |

Review conclusion: the implementation addresses the mechanism and logging risk. It does not yet provide real evidence that the learned residual is a true age trajectory.

## 4. Active Config Set

Status: PASS

The active branch keeps only the retained ACSM v1/v2/v3 configs. Removed
ablation configs should not be referenced by new experiments.

Smoke command:

```bash
/xmudata/pzj/envs/casv1/bin/python -c "exec(\"import yaml, glob\nfrom wespeaker.models.speaker_model import get_speaker_model\nfrom wespeaker.models.acsm_modules import get_acsm_config\npaths=sorted(glob.glob('examples/voxceleb/v2/conf/resnet34_acsm*.yaml'))\nfor p in paths:\n    cfg=yaml.safe_load(open(p))\n    cfg['model_args']['acsm_args']=get_acsm_config(cfg)\n    model=get_speaker_model(cfg['model'])(**cfg['model_args'])\n    print(p, type(model).__name__)\")"
```

Result: PASS for all ACSM configs.

## 5. Fair Evaluation

Status: PARTIAL

Script: `wespeaker_acsm/tools/audit_fair_eval.py`

Example command:

```bash
/xmudata/pzj/envs/casv1/bin/python tools/audit_fair_eval.py \
  --config examples/voxceleb/v2/conf/resnet34_acsm.yaml \
  --score-py wespeaker/bin/score.py \
  --trial-list examples/voxceleb/v2/data/baseline/trials/vox1_O_cleaned.kaldi
```

Checks:

| Check | Status | Evidence |
| --- | --- | --- |
| Extraction does not read true age label | PASS | Extraction path does not pass `age_group`; audit flags top-level extraction age-label requirements. |
| `score.py` does not read age label | PASS | Audit checks `score.py`; no age-label read is detected. |
| ACSM default uses predicted posterior | PASS | Forward works without `age_group`; age posterior comes from observer. |
| Oracle age default disabled | PASS | No default oracle-age eval path is enabled. |
| Oracle mode must mark `oracle_age_used=True` | PARTIAL | Audit checks for this convention if oracle options are present, but oracle eval is not a first-class implemented path. |
| Audit script exists | PASS | `tools/audit_fair_eval.py`. |
| Fair eval report generated | PARTIAL | Audit output was generated in command output and summarized in docs; no dedicated committed JSON audit artifact is required by the current code path. |

Review conclusion: the normal evaluation path is fair with respect to test age. The remaining weakness is that oracle-age handling is enforced by audit convention rather than by a central runtime guard.

## 6. Data Leakage

Status: PARTIAL

Script: `wespeaker_acsm/tools/audit_data_leakage.py`

Example command:

```bash
/xmudata/pzj/envs/casv1/bin/python tools/audit_data_leakage.py \
  --train-utt2spk examples/voxceleb/v2/data/baseline/vox2_train_voxca/utt2spk \
  --eval-utt-list examples/voxceleb/v2/data/baseline/vox1/wav.scp \
  --age-label-file /xmudata/pzj/vox-ca/vox2dev/segment2age.npy \
  --eval-trial examples/voxceleb/v2/data/baseline/trials/vox1_O_cleaned.kaldi
```

Checks:

| Check | Status | Evidence |
| --- | --- | --- |
| Train/eval speaker overlap | PASS | Script checks and reports overlap count/examples. |
| Train/eval utterance overlap | PASS | Script checks and reports overlap count/examples. |
| Age-label keys vs eval trial keys | PARTIAL | Script checks exact-key intersection, but does not fully normalize all candidate Vox key forms, so it can undercount label coverage. |
| Trial utterance existence | PASS | Script validates trial utterances against eval utt list. |
| Positive/negative counts | PASS | Script reports target/nontarget counts. |
| JSON report | PASS | `--output-json` writes a JSON report. |
| Distinguishes label existence from label use | PASS | Report text distinguishes age labels present on eval utterances from leakage through model use. |

Observed full audit on Vox-CA/official trial files reported no speaker overlap, no utterance overlap, no invalid trial labels, no self-trials, and no missing eval utterances. The age-label overlap result should be treated cautiously because of exact-key matching.

## 7. Batch Pair Coverage

Status: PASS

Script: `wespeaker_acsm/tools/diagnose_age_pair_coverage.py`

Example command:

```bash
/xmudata/pzj/envs/casv1/bin/python tools/diagnose_age_pair_coverage.py \
  --utt2spk examples/voxceleb/v2/data/baseline/vox2_train_voxca/utt2spk \
  --age-label-file /xmudata/pzj/vox-ca/vox2dev/segment2age.npy \
  --age-label-type value \
  --batch-size 96 --num-batches 20
```

Result: PASS. The script reports valid-age sample count, same-speaker different-age pair count, `path_nonzero_batch_ratio`, and a recommendation. No sampler was rewritten or enabled by default.

Observed smoke estimate: `path_nonzero_batch_ratio` was about `0.55` for the sampled setup, with low average pair count. This supports weak path ablation but still argues against treating path loss as a primary constraint without more sampling analysis.

## 8. Checkpoint Loading

Status: PASS

Checks:

| Check | Status | Evidence |
| --- | --- | --- |
| ResNet34 checkpoint partial-loads into ACSM | PASS | Implemented in `wespeaker/utils/checkpoint.py` with `allow_acsm_partial`. |
| `strict=true` keeps strict behavior | PASS | Strict branch calls `load_state_dict(..., strict=True)`. |
| `strict=false` only allows ACSM missing keys | PASS | Non-ACSM missing keys raise when ACSM partial load is enabled. |
| Load report | PASS | Reports loaded, missing, unexpected, ACSM-missing counts and examples. |
| Test coverage | PASS | `tests/test_acsm.py` includes fake checkpoint partial-load and strict behavior tests. |

Review conclusion: this is one of the stronger fixes. It prevents silently accepting unrelated checkpoint mismatches while supporting the intended baseline-to-ACSM initialization.

## 9. Ordinary SV Protection

Status: PARTIAL

Checks:

| Check | Status | Evidence |
| --- | --- | --- |
| Consistency loss retained | PASS | `lambda_consistency` remains in configs/loss. |
| v2/v3 configs exist | PASS | `resnet34_acsm_main.yaml`, `resnet34_acsm_main_v3.yaml`. |
| Logs gate/residual/raw-can distance | PASS | Executor logs gate, residual norm, cosine, L2 distance, and path pair count. |
| Docs require ordinary SV reporting | PASS | Experiment plan and claim guideline require ordinary SV metrics. |

Review conclusion: the protection mechanisms and measurements exist, but ordinary SV preservation is not proven until Vox-O/H or comparable ordinary SV metrics are run.

## 10. Model Profile

Status: PASS

Script: `wespeaker_acsm/tools/profile_model.py`

Example command:

```bash
/xmudata/pzj/envs/casv1/bin/python tools/profile_model.py \
  --config examples/voxceleb/v2/conf/resnet34_acsm.yaml \
  --include-baseline --device cpu --batch-size 1 --frames 80 --warmup 1 --iters 2
```

Checks:

| Check | Status | Evidence |
| --- | --- | --- |
| Script exists | PASS | `tools/profile_model.py`. |
| Parameter count | PASS | Reports total and trainable params. |
| ACSM extra params | PASS | Reports ACSM extra params when baseline comparison is requested. |
| Latency | PASS | Measures forward latency on CPU/GPU when requested and available. |
| Baseline vs ACSM comparison | PASS | `--include-baseline` constructs and profiles baseline. |
| FLOPs honesty | PASS | FLOPs are `null` with an explicit note when no FLOPs dependency is used. |

Observed smoke output included baseline params around 6.63M and ACSM params around 6.68M, with ACSM extra params around 50K.

## 11. Experiment Plan

Status: PASS

Document: `wespeaker_acsm/docs/ACSM_EXPERIMENT_PLAN.md`

Checks:

| Check | Status |
| --- | --- |
| Main experiment table | PASS |
| Ablation table | PASS |
| EER/minDCF metrics | PASS |
| Parameter/latency metrics | PASS |
| Multi-seed plan | PASS |
| Ordinary SV and CA gap tests | PASS |
| Result saving/naming rules | PASS |

Review conclusion: the plan is adequate for avoiding premature claims, provided future runs follow it.

## 12. Claim Guidelines

Status: PASS

Document: `wespeaker_acsm/docs/ACSM_CLAIM_GUIDELINES.md`

Checks:

| Check | Status |
| --- | --- |
| Forbids over-strong claims | PASS |
| States evidence required per claim | PASS |
| Says no real EER means no performance claim | PASS |
| Requires multi-seed/significance for small gains | PASS |
| Requires downscoping if ordinary SV regresses | PASS |

Review conclusion: the claim policy is explicit and publication-safe if followed.

## 13. Reproducibility

Status: PARTIAL

Checks:

| Check | Status | Evidence |
| --- | --- | --- |
| Records random seed | PASS | Training config/manifest include seed. |
| Saves config | PASS | Training writes `config.yaml`. |
| Saves git commit hash | PASS | Manifest includes git commit hash. |
| Saves experiment manifest | PASS | `experiment_manifest.yaml` is written by training. |
| Records `age_label_file` | PASS | Manifest captures config values including ACSM age-label path. |
| Records ACSM hparams | PASS | Manifest captures model/config fields including ACSM loss and module settings. |
| Avoids overwriting old results | PARTIAL | Documentation instructs unique result naming; hard runtime protection is not enforced. |
| DataLoader worker seed control | PARTIAL | Global seed exists, but explicit worker seeding was not deeply audited or newly standardized. |
| Dirty worktree traceability | PARTIAL | Manifest records commit hash, but uncommitted local changes can still make the hash incomplete. |

Small real-training smoke:

```bash
/xmudata/pzj/envs/casv1/bin/torchrun --standalone --nproc_per_node=1 \
  -m wespeaker.bin.train --config /tmp/acsm_train_smoke_config/acsm_train_smoke.yaml
```

Result: PASS. One epoch, two batches, fake-small real loader smoke completed and wrote:

- `/tmp/acsm_train_smoke/models/model_1.pt`
- `/tmp/acsm_train_smoke/models/final_model.pt`
- `/tmp/acsm_train_smoke/config.yaml`
- `/tmp/acsm_train_smoke/experiment_manifest.yaml`

The smoke log included ACSM diagnostics and nonzero `path_valid_pair_count`. This validates plumbing only; it is not evidence of performance improvement.

## Additional Verification Commands

Syntax check:

```bash
/xmudata/pzj/envs/casv1/bin/python -m py_compile \
  wespeaker/models/acsm_modules.py \
  wespeaker/models/resnet.py \
  wespeaker/models/speaker_model.py \
  wespeaker/utils/checkpoint.py \
  wespeaker/bin/train.py \
  wespeaker/bin/extract.py \
  wespeaker/utils/executor.py \
  tools/diagnose_acsm.py \
  tools/audit_fair_eval.py \
  tools/audit_data_leakage.py \
  tools/diagnose_age_pair_coverage.py \
  tools/profile_model.py \
  tests/test_acsm.py
```

Result: PASS.

ACSM fake-batch risk smoke:

```text
ACSM_RISK_SMOKE_OK
loss_total 1.93333
path_valid_pair_count 1.0
cos_raw_can_mean 1.0
l2_raw_can_mean 0.0
```

Interpretation: loss/backward plumbing works and path pairs are counted; the near-identity raw/canonical output confirms why diagnostics are required.

## Remaining Risks

1. `pytest` is not installed in the active environment, so the exact requested pytest command has not passed.
2. ACSM activity is diagnosable but not proven; smoke runs still look close to identity, which is expected before real training.
3. Canonical trajectory claims remain unsupported until `lambda_path=0` vs `lambda_path>0` ablations are run with sufficient cross-age pair coverage.
4. Oracle-age behavior is prevented in the default path, but stronger runtime guards would be needed if oracle evaluation becomes a first-class feature.
5. Age-label overlap audit should normalize Vox key variants before being treated as exhaustive.
6. Ordinary SV preservation is not established; it needs real Vox-O/H or equivalent evaluation.
7. Reproducibility is good enough for smoke tests, but dirty-worktree recording and worker-seed control should be tightened before final experiments.

## First Real Training Recommendation

Recommended entry point: run a small real-data stability training with the current ACSM v3 configuration first. This should be treated as a training-pipeline smoke test, not as a result-producing experiment.

```bash
cd /xmudata/pzj/casv/wespeaker_acsm
/xmudata/pzj/envs/casv1/bin/torchrun --standalone --nproc_per_node=1 \
  -m wespeaker.bin.train \
  --config examples/voxceleb/v2/conf/resnet34_acsm_main_v3.yaml
```

If runtime is a concern, first use the already validated small smoke config pattern from `/tmp/acsm_train_smoke_config/acsm_train_smoke.yaml`, then move to `resnet34_acsm_main_v3.yaml` with a uniquely named experiment directory. Do not overwrite previous results.

## Recommendation

Proceed to controlled small-scale real training only as a plumbing and stability check. Do not report ACSM as improving EER, learning a true age trajectory, or preserving ordinary SV until the planned multi-seed experiments, ordinary SV evaluation, cross-age trial evaluation, diagnostics, and significance checks are completed.

Passing this review means the risk controls are mostly in place. It does not mean ACSM improves EER, reduces minDCF, preserves ordinary SV, or learns a real age trajectory.
