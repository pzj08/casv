# ACSM Risk Fix Report

## Summary

This pass reviewed and hardened the current ACSM-Net / `ResNet34_ACSM`
implementation without adding a new major modeling method and without running
formal training. The changes target correctness, fair evaluation, leakage
auditing, diagnostic visibility, partial checkpoint initialization, and
experiment discipline.

No EER or minDCF improvement is claimed.

## Risk Status

| Risk | Status | Fix |
| --- | --- | --- |
| A. Missing ACSM unit tests | FIXED | Expanded `tests/test_acsm.py` for modules, forward, losses, path pairs, partial checkpoint loading, and baseline compatibility. |
| B. ACSM may be an empty module | FIXED | Added `tools/diagnose_acsm.py` and train diagnostics for gate, residual, raw/canonical cosine and L2. |
| C. Weak trajectory evidence | PARTIALLY FIXED | Added path pair counts and effectiveness diagnostics. Real evidence still requires checkpoint-level diagnostics and official evaluation. |
| D. AgeFiLM necessity unproven | PARTIALLY FIXED | Diagnostics can inspect behavior, but ablation configs were removed from the active branch. |
| E. Fair evaluation and test-age leakage | FIXED | Added `tools/audit_fair_eval.py`; extraction still uses predicted posterior and `score.py` remains age-free. |
| F. Speaker/utterance/trial/age label leakage | FIXED | Added `tools/audit_data_leakage.py`; local Vox-CA audit found no train/eval speaker or utterance overlap. |
| G. Batch sampling may make path loss zero | PARTIALLY FIXED | Added path pair diagnostics and `tools/diagnose_age_pair_coverage.py`; sampler not changed by default. |
| H. Baseline checkpoint into ACSM | FIXED | Added `model_init_strict`, ACSM-aware partial loading report, and tests. |
| I. Ordinary ASV degradation risk | PARTIALLY FIXED | Kept consistency loss, added raw/canonical diagnostics plus safe/aggressive configs. Real Vox-O/E/H results still required. |
| J. Complexity reporting missing | FIXED | Added `tools/profile_model.py` for params and latency; FLOPs are not computed to avoid new dependencies. |
| K. Experiment matrix missing | FIXED | Added `docs/ACSM_EXPERIMENT_PLAN.md`. |
| L. Claims too strong | FIXED | Added `docs/ACSM_CLAIM_GUIDELINES.md`. |
| M. Reproducibility metadata | PARTIALLY FIXED | Training now writes `experiment_manifest.yaml`; DataLoader worker seed policy remains inherited from existing WeSpeaker pipeline. |

## Modified Files

- `wespeaker/models/acsm_modules.py`
- `wespeaker/models/resnet.py`
- `wespeaker/utils/executor.py`
- `wespeaker/utils/checkpoint.py`
- `wespeaker/bin/train.py`
- `tests/test_acsm.py`
- `docs/ACSM_IMPLEMENTATION_REPORT.md`

## New Files

- `tools/diagnose_acsm.py`
- `tools/audit_fair_eval.py`
- `tools/audit_data_leakage.py`
- `tools/diagnose_age_pair_coverage.py`
- `tools/profile_model.py`
- `examples/voxceleb/v2/conf/resnet34_acsm_main_v3.yaml`
- `docs/ACSM_EFFECTIVENESS_DIAGNOSTICS.md`
- `tests/test_acsm_effectiveness_diagnostics.py`
- `docs/ACSM_EXPERIMENT_PLAN.md`
- `docs/ACSM_CLAIM_GUIDELINES.md`
- `docs/ACSM_RISK_FIX_REPORT.md`

## Tests And Checks Run

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

```bash
/xmudata/pzj/envs/casv1/bin/python -m unittest tests.test_acsm
```

Result: PASS, 26 tests.

```bash
/xmudata/pzj/envs/casv1/bin/python -m pytest tests/test_acsm.py tests/test_acsm_effectiveness_diagnostics.py -q
```

Result: NOT RUN in this environment because `pytest` is not installed.

```bash
/xmudata/pzj/envs/casv1/bin/python tools/audit_fair_eval.py \
  --config examples/voxceleb/v2/conf/resnet34_acsm.yaml \
  --score-py wespeaker/bin/score.py \
  --trial-list examples/voxceleb/v2/data/baseline/trials/vox1_O_cleaned.kaldi
```

Result: PASS.

```bash
/xmudata/pzj/envs/casv1/bin/python tools/audit_data_leakage.py \
  --train-utt2spk examples/voxceleb/v2/data/baseline/vox2_train_voxca/utt2spk \
  --eval-utt-list examples/voxceleb/v2/data/baseline/vox1/wav.scp \
  --age-label-file /xmudata/pzj/vox-ca/vox2dev/segment2age.npy \
  --eval-trial ... all Vox-CA/MIM trial files ...
```

Result: PASS. Train/eval speaker overlap: 0. Train/eval utterance overlap: 0.
Trial label/self-trial/missing-utterance checks: PASS.

```bash
/xmudata/pzj/envs/casv1/bin/python tools/diagnose_age_pair_coverage.py \
  --utt2spk examples/voxceleb/v2/data/baseline/vox2_train_voxca/utt2spk \
  --age-label-file /xmudata/pzj/vox-ca/vox2dev/segment2age.npy \
  --age-label-type value \
  --batch-size 96 \
  --num-batches 20
```

Result: PASS. Random-batch nonzero path-pair ratio was 0.55 in this smoke
sample.

```bash
/xmudata/pzj/envs/casv1/bin/python tools/profile_model.py \
  --config examples/voxceleb/v2/conf/resnet34_acsm.yaml \
  --include-baseline \
  --device cpu \
  --batch-size 1 \
  --frames 80 \
  --warmup 1 \
  --iters 2
```

Result: PASS. Baseline params: 6,634,336. ACSM params: 6,684,520. ACSM extra
params: 50,184. CPU latency was measured only as a smoke check.

Fake-batch ACSM smoke result: PASS. It checked finite total loss, finite
embeddings/posteriors, gate range, path-valid pair count, raw/canonical
diagnostics, and gradients for age observer, AgeFiLM, transition, and gate
parameters.

## Small Real Training Smoke

Command:

```bash
/xmudata/pzj/envs/casv1/bin/torchrun --standalone --nproc_per_node=1 \
  -m wespeaker.bin.train \
  --config /tmp/acsm_train_smoke_config/acsm_train_smoke.yaml
```

Scope:

- Data: `examples/voxceleb/v2/data/baseline/smoke_train/raw.list`
- Labels: `examples/voxceleb/v2/data/baseline/smoke_train/utt2spk`
- Age groups: `examples/voxceleb/v2/data/baseline/smoke_train/age_groups`
- Epochs: 1
- Iterations: 2
- Batch size: 4
- Output directory: `/tmp/acsm_train_smoke`

Result: PASS. The run completed, wrote `model_1.pt`, `final_model.pt`,
`config.yaml`, `experiment_manifest.yaml`, and logged ACSM diagnostics. The
logged path-valid pair count was 6.0 for both smoke batches. This confirms the
training path executes and saves checkpoints; it is not a performance
experiment.

## Still Open

- No formal training has been run.
- No EER/minDCF result exists for ACSM.
- No multi-seed or statistical significance analysis has been run.
- FLOPs/MACs are not computed; `profile_model.py` reports params and latency
  only.
- DataLoader worker seeding remains the existing WeSpeaker behavior.
- Path loss coverage is estimated under random batches only; no age-aware
  sampler was added.

## Recommendation

Proceed to a small real training smoke only after reviewing the new audits.
Recommended first experiment:

1. `resnet34_acsm.yaml` as the retained v1 reference.
2. `resnet34_acsm_main.yaml` as the v2 comparison point.
3. `resnet34_acsm_main_v3.yaml` as the current candidate.

Performance conclusions must wait for real Vox-CA/MIM evaluation. This risk
fix pass does not show that ACSM improves EER.
