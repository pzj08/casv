# ACSM Pre-Experiment Validation Report

Date: 2026-06-26

Purpose: validate ACSM-Net code correctness, workflow completeness, evaluation fairness, data-leakage readiness, diagnostic readiness, and experiment readiness before formal training/evaluation. This report does not claim ACSM improves EER, minDCF, ordinary SV, or cross-age SV.

## Final Decision

**CONDITIONAL_GO**

The ACSM code path, tests, tiny train/extract/score/metrics smoke, fairness audit, data leakage audit, trajectory diagnosis, and profile tooling are ready enough to start controlled small-scale and then formal experiments. The main condition is reproducibility: the worktree is currently dirty and includes uncommitted ACSM-related files, so the current git commit hash alone is not a complete experiment identifier. Commit or otherwise archive the exact diff before official runs.

## Git And Environment

Command:

```bash
git rev-parse HEAD
```

Result:

```text
3df63fff01edc14f2527657dc3099f73c9b73ad3
```

Command:

```bash
git status --short
```

Result: dirty worktree. Important ACSM-related dirty/untracked files include:

```text
 M docs/ACSM_IMPLEMENTATION_REPORT.md
 M tests/test_acsm.py
 M tools/profile_model.py
 M wespeaker/models/acsm_modules.py
 M wespeaker/models/resnet.py
 M wespeaker/utils/executor.py
?? docs/ACSM_TRAJECTORY_AND_PARAMMATCH_REPORT.md
?? examples/voxceleb/v2/conf/resnet34_acsm_main_v3.yaml
?? examples/voxceleb/v2/conf/resnet34_parammatch.yaml
?? tests/test_acsm_trajectory_diagnosis.py
?? tools/diagnose_acsm_trajectory.py
```

There are also unrelated dirty/untracked files outside this ACSM validation scope, including baseline config edits and local prompt/auxiliary directories. Because of this, `3df63fff...` must not be used alone as the full reproducibility identifier.

Environment:

```text
Python 3.9.21
torch 2.5.1+cu121
torchaudio 2.5.1+cu121
yaml/numpy ok
pytest installed and usable
```

## Code Correctness

Command:

```bash
python -m py_compile \
  wespeaker/models/acsm_modules.py \
  wespeaker/models/resnet.py \
  wespeaker/models/speaker_model.py \
  wespeaker/utils/checkpoint.py \
  wespeaker/bin/train.py \
  wespeaker/bin/extract.py \
  wespeaker/bin/score.py \
  wespeaker/utils/executor.py \
  tools/audit_fair_eval.py \
  tools/audit_data_leakage.py \
  tools/diagnose_acsm.py \
  tools/diagnose_age_pair_coverage.py \
  tools/profile_model.py \
  tools/diagnose_acsm_trajectory.py \
  tests/test_acsm_trajectory_diagnosis.py
```

Result: **PASS**.

`score.py` compiled and fairness audit confirms it does not read age labels.

## Unit Tests

Command:

```bash
python -m pytest tests/test_acsm.py tests/test_acsm_effectiveness_diagnostics.py -q
```

Result:

```text
32 passed, 3 warnings
```

Command:

```bash
python -m pytest tests/test_acsm_trajectory_diagnosis.py -q
```

Result:

```text
4 passed, 2 warnings
```

Covered checks:

- ACSM module tests pass.
- `AgeFiLM2d` identity/bypass behavior passes.
- `Stage2AgeObserver` posterior/loss behavior passes.
- `OrderedAgeCanonicalizer` reference residual and identity behavior pass.
- `PathConsistencyLoss` valid-pair/no-pair/ignore-age behavior passes.
- `ResNet34_ACSM` forward/backward passes.
- Baseline `ResNet34` tuple output remains unchanged.
- Checkpoint partial loading is covered.
- Trajectory diagnosis embedding/model modes are covered.

## Model Build Closure

Command: model build/forward smoke for `ResNet34` and `ResNet34_ACSM`.

Result:

```text
MODEL ResNet34 <function ResNet34 ...>
ResNet34 output type: <class 'tuple'>
MODEL ResNet34_ACSM <function ResNet34_ACSM ...>
ACSM keys: ['acsm_loss_inputs', 'age_embedding', 'age_posterior', 'age_pred',
            'canonical_residual', 'embedding', 'gate', 'path_norm',
            'rank_logits', 'raw_embedding', 'uncertainty']
MODEL_BUILD_AND_FORWARD_OK
```

Status: **PASS**.

Notes:

- `get_speaker_model("ResNet34_ACSM")` succeeds.
- ACSM is a structural ResNet34 variant with no legacy wrapper dependency.
- Forward succeeds without `age_group`.
- `embedding` is canonical embedding.
- `raw_embedding` is observed embedding.
- Diagnostic fields are present.

## Forward/Backward Smoke

Command: fake batch backward with `compute_acsm_losses`.

Result:

```text
GRAD_PARAM age_observer.raw_delta
ACSM_BACKWARD_OK
```

Status: **PASS**.

Confirmed:

- ACSM loss finite.
- Backward succeeds.
- ACSM-related parameter receives gradient.
- No NaN/Inf observed.

## Baseline Compatibility

Status: **PASS**.

Evidence:

- Unit tests confirm `ResNet34` still returns tuple output.
- `score.py` unchanged in protocol and does not read age labels.
- `extract.py` handles dict output by taking `outputs["embedding"]` and handles tuple output for baseline.

## Configuration Completeness

Command: load all ACSM configs under `examples/voxceleb/v2/conf/resnet34_acsm*.yaml`.

Result: **PASS**.

Configs found and constructed:

- `resnet34_acsm.yaml`
- `resnet34_acsm_main.yaml`
- `resnet34_acsm_main_v3.yaml`

Required coverage:

| Requirement | Status |
| --- | --- |
| Stable `lambda_path=0` config | PASS |
| Main `lambda_path=0.01` config | PASS |
| Path strength `lambda_path=0.02` config | PASS |
| Safe config | PASS |
| Aggressive config | PASS |
| w/o AgeFiLM | PASS |
| w/o canonicalizer | PASS |
| w/o age loss | PASS |
| w/o consistency | PASS |

## Train/Extract/Score Smoke

Train smoke command:

```bash
torchrun --standalone --nproc_per_node=1 \
  -m wespeaker.bin.train \
  --config /tmp/acsm_preexp_train_smoke_config/config.yaml
```

Result: **PASS**. One epoch, two batches.

The first attempt using the old `/tmp/acsm_train_smoke` output directory failed because the directory already existed and resume logic expected a checkpoint. Re-running with a fresh `/tmp/acsm_preexp_train_smoke` output directory passed.

Training log confirms:

```text
loss_age
loss_consistency
loss_smooth
loss_path
loss_acsm_total
path_valid_pair_count
path_nonzero_batch_ratio
gate_mean
residual_norm_mean
cos_raw_can_mean
raw_can_cosine_mean
```

Example log line:

```text
ACSM loss_spk=25.299557, loss_age=0.553003,
loss_consistency=31.667706, loss_smooth=0.003065,
loss_path=0.033546, loss_acsm_total=0.688990,
gate_mean=0.000788, residual_norm_mean=0.031913,
cos_raw_can_mean=1.000000, raw_can_cosine_mean=1.000000,
path_valid_pair_count=6.000000, path_nonzero_batch_ratio=1.000000
```

Extract smoke command:

```bash
python -m wespeaker.bin.extract --config /tmp/acsm_preexp_extract/config.yaml
```

Result: **PASS**. Wrote 8 embeddings to `/tmp/acsm_preexp_extract/xvector.scp`.

Confirmed:

- Extraction does not pass true age labels.
- Dict output uses `outputs["embedding"]`.

Score smoke command:

```bash
python -m wespeaker.bin.score \
  /tmp/acsm_preexp_score \
  /tmp/acsm_preexp_extract/xvector.scp \
  False \
  /tmp/acsm_preexp_extract \
  /tmp/acsm_preexp_extract/toy.trials
```

Result: **PASS**. Wrote `/tmp/acsm_preexp_score/scores/toy.trials.score`.

Metrics smoke command:

```bash
python -m wespeaker.bin.compute_metrics \
  0.01 1 1 /tmp/acsm_preexp_score/scores/toy.trials.score
```

Result: **PASS**.

```text
---- toy.trials.score -----
EER = 0.000
minDCF (p_target:0.01 c_miss:1 c_fa:1) = 0.000
```

This toy score is only a pipeline smoke result and must not be interpreted as model performance.

## Fair Evaluation Audit

Help command:

```bash
python tools/audit_fair_eval.py --help
```

Result: **PASS**.

Audit command:

```bash
python tools/audit_fair_eval.py \
  --config examples/voxceleb/v2/conf/resnet34_acsm_main.yaml \
  --score-py wespeaker/bin/score.py \
  --trial-list examples/voxceleb/v2/data/baseline/trials/vox1_O_cleaned.kaldi \
  --embedding-scp /tmp/acsm_preexp_extract/xvector.scp \
  --output-json /tmp/acsm_preexp_fair_eval.json
```

Result:

```text
status: PASS
```

Confirmed:

- ACSM uses predicted age posterior by default.
- Extraction config does not pass `age_group`.
- No top-level extraction `age_label_file` is required.
- Oracle age is disabled by default.
- `score.py` does not reference age labels.
- Trial list exists.
- Embedding scp has key/path columns only.

## Data Leakage Audit

Help command:

```bash
python tools/audit_data_leakage.py --help
```

Result: **PASS**.

Real-path audit command:

```bash
python tools/audit_data_leakage.py \
  --train-utt2spk examples/voxceleb/v2/data/baseline/vox2_train_voxca/utt2spk \
  --eval-utt-list examples/voxceleb/v2/data/baseline/vox1/wav.scp \
  --eval-trial examples/voxceleb/v2/data/baseline/trials/vox1_O_cleaned.kaldi \
  --age-label-file /xmudata/pzj/vox-ca/vox2dev/segment2age.npy \
  --output-json /tmp/acsm_preexp_data_leakage.json
```

Result:

```text
status: PASS
train_speaker_count: 5990
eval_speaker_count: 1251
train_eval_speaker_overlap_count: 0
train_eval_utt_overlap_count: 0
invalid_label_count: 0
self_trial_count: 0
missing_eval_utt_count: 0
positive_count: 18802
negative_count: 18809
```

Note: the age-label overlap check is exact-key based. It is useful as an audit signal, but final reporting should still rely on the fair-eval audit to ensure age labels are not used by extraction/scoring.

## Trajectory Diagnosis Readiness

Help command:

```bash
python tools/diagnose_acsm_trajectory.py --help
```

Result: **PASS**.

Confirmed:

- Supports embedding mode.
- Supports model mode.
- Supports `--wav-scp`.
- Supports `--data-list --data-type raw/feat`.
- Supports `--train-log`.
- Does not pass true `age_group` to model forward.
- Uses age labels only for grouping/statistics.
- Outputs `oracle_age_used=false`.
- Model mode outputs real `gate_mean`, `residual_norm_mean`, `uncertainty_mean`, and `raw_can_cosine_mean` without external diagnostic JSON.

Model-mode smoke command:

```bash
python tools/diagnose_acsm_trajectory.py \
  --mode model \
  --config /tmp/acsm_traj_model_smoke/config.yaml \
  --checkpoint /tmp/acsm_traj_model_smoke/model.pt \
  --wav-scp /tmp/acsm_traj_model_smoke/wav.scp \
  --utt2spk /tmp/acsm_traj_model_smoke/utt2spk \
  --age-label-file /tmp/acsm_traj_model_smoke/age.npy \
  --batch-size 2 \
  --device cpu \
  --max-utts 4 \
  --max-pairs 4 \
  --train-log /tmp/acsm_preexp_train_smoke/train.log \
  --output-json /tmp/acsm_preexp_trajectory.json
```

Result: **PASS**.

Key output:

```text
oracle_age_used: false
gate_mean: 0.00047785886272322387
residual_norm_mean: 0.015209715813398361
uncertainty_mean: 0.9919824302196503
raw_can_cosine_mean: 1.0
path_nonzero_batch_ratio: 1.0
unavailable_metrics: {}
```

## ACSM Diagnostics Readiness

Command:

```bash
python tools/diagnose_acsm.py \
  --config /tmp/acsm_preexp_train_smoke/config.yaml \
  --checkpoint /tmp/acsm_preexp_train_smoke/models/final_model.pt \
  --fake \
  --batch-size 4 \
  --num-frames 80 \
  --device cpu \
  --output-json /tmp/acsm_preexp_diagnose_acsm.json
```

Result: **PASS**.

Key output:

```text
gate_mean: 0.0006617161561734974
residual_norm_mean: 0.03187689185142517
raw_can_cosine_mean: 1.0
path_norm_by_age_group: available
transition_norms: available
```

Interpretation: the smoke checkpoint is near identity, which is expected for tiny/untrained smoke and not a performance claim.

Age-pair coverage command:

```bash
python tools/diagnose_age_pair_coverage.py \
  --utt2spk examples/voxceleb/v2/data/baseline/vox2_train_voxca/utt2spk \
  --age-label-file /xmudata/pzj/vox-ca/vox2dev/segment2age.npy \
  --age-label-type value \
  --batch-size 96 \
  --num-batches 20
```

Result:

```text
path_nonzero_batch_ratio: 0.55
path_valid_pair_count_mean: 0.55
recommendation: Random batches have enough path pairs for weak lambda_path ablation.
```

## Profile / Parameter Readiness

Command:

```bash
python tools/profile_model.py \
  --config examples/voxceleb/v2/conf/resnet34_acsm_main.yaml \
  --include-baseline \
  --include-parammatch \
  --device cpu \
  --batch-size 1 \
  --frames 200 \
  --warmup 2 \
  --iters 5 \
  --output-json /tmp/acsm_preexp_profile.json
```

Result: **PASS**.

Key output:

| Model | Total params | Extra over ResNet34 | CPU latency ms |
| --- | ---: | ---: | ---: |
| ResNet34 | 6,634,336 | 0 | 26.01 |
| ResNet34_ACSM | 6,684,520 | 50,184 | 34.49 |
| ResNet34_ParamMatch | 6,684,705 | 50,369 | 32.23 |

FLOPs/MACs are not computed; the profile script explicitly reports FLOPs unavailable to avoid adding dependencies.

## Claim-Evidence Readiness

| Claim | Required evidence | Current readiness | Status |
| --- | --- | --- | --- |
| ACSM is an architecture-level ResNet variant | Code inspection, factory construction, baseline tests | `ResNet34_ACSM` is a ResNet structural variant | READY_TO_TEST |
| ACSM forward uses predicted age posterior | Forward without `age_group`, fair audit, trajectory model mode | Forward succeeds without age labels; `oracle_age_used=false` | READY_TO_TEST |
| AgeFiLM is necessary | w/o AgeFiLM ablation config and real results | Config exists, no real result yet | READY_TO_TEST |
| OrderedAgeCanonicalizer is necessary | w/o canonicalizer ablation config and real results | Config exists, no real result yet | READY_TO_TEST |
| Path consistency supports trajectory claim | `lambda_path=0/0.01/0.02` ablation plus trajectory diagnostics | Configs and diagnostics exist; no real ablation result yet | PARTIALLY_READY |
| ACSM is not only parameter-count gain | ParamMatch model/profile and real result comparison | ParamMatch exists and profiles close to ACSM; no real result yet | READY_TO_TEST |
| ACSM does not use true test age | Fair audit, extraction path, score path | Audit PASS; extraction/scoring age-free | READY_TO_TEST |
| ACSM does not significantly damage ordinary SV | Ordinary SV EER/minDCF results | No real ordinary SV result yet | READY_TO_TEST |
| ACSM improves cross-age SV | Vox-CA cross-age EER/minDCF | No real cross-age result yet | READY_TO_TEST |
| ACSM improvement is statistically reliable | Multi-seed and bootstrap/significance analysis | Planned only | NOT_READY |

No performance claim is supported yet.

## Formal Experiment Blockers

No code-level **NO_GO** blockers were found:

- `ResNet34_ACSM` builds.
- Forward passes.
- Backward passes.
- Unit tests pass.
- Baseline `ResNet34` behavior is not broken.
- Extraction does not require true age labels.
- `score.py` does not depend on age labels.
- Legacy removed modules are not part of the active code path.
- ACSM configs load.
- Checkpoint partial loading is tested.
- No speaker/utterance/trial leakage was found in the audited Vox-O trial setup.

Reproducibility condition before official runs:

- Commit or otherwise archive the dirty worktree. The current commit hash alone is not sufficient.

## Non-Blocking Risks

- Tiny smoke train/extract/score checks verify plumbing only, not training stability at scale.
- The smoke checkpoint is near identity; this is not a failure, but real training diagnostics must be monitored.
- Data leakage audit reports age-label overlap by exact key; fair-eval audit remains the decisive check that age labels are not used at extraction/scoring.
- `path_nonzero_batch_ratio` from random batches is moderate; if real training has low path-pair coverage, an age-aware sampler may be needed later.
- FLOPs are not computed, only params and latency.
- Statistical reliability is not ready until multi-seed and bootstrap/significance tests are implemented or run.

## Recommended First Formal Experiments

Run in this order after committing/archiving the exact code state:

1. ResNet34 official baseline.
2. ACSM v1, `resnet34_acsm.yaml`, only as retained reference.
3. ACSM v2, `resnet34_acsm_main.yaml`.
4. ACSM v3, `resnet34_acsm_main_v3.yaml`.
5. ParamMatch, if resources allow.
6. Effectiveness diagnostics for every ACSM checkpoint used in reporting.

For every experiment save:

- config;
- full git commit hash plus dirty diff status if any;
- seed;
- checkpoint;
- score files;
- EER/minDCF;
- gate/residual/trajectory diagnostics;
- data leakage audit result;
- fair evaluation audit result.

Allowed wording before real results:

- “ACSM code path passed pre-experiment validation.”
- “ACSM is ready for formal experiments under the recorded protocol.”
- “ACSM claims have a testable experimental design.”
- “Performance improvement requires formal EER/minDCF confirmation.”

Disallowed wording before real results:

- “ACSM improves EER.”
- “ACSM outperforms baseline.”
- “ACSM learned a real age trajectory.”
- “ACSM does not damage ordinary SV.”
- “ACSM is proven effective.”
