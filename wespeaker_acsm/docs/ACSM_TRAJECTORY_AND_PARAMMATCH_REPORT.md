# ACSM Trajectory and Parameter-Matched Control Report

## Purpose

ACSM cannot support a canonical trajectory claim only by containing an ordered
canonicalizer. The implementation must be checked with speaker/age-aware
diagnostics that show whether canonicalization actually reduces same-speaker
cross-age distance, while not also reducing different-speaker distance in a way
that suggests embedding collapse.

This report documents two research validation tools:

- `tools/diagnose_acsm_trajectory.py`
- `ResNet34_ParamMatch`

No EER or minDCF improvement is claimed here.

## Canonical Trajectory Diagnostic

Script:

Embedding mode:

```bash
python tools/diagnose_acsm_trajectory.py \
  --mode embedding \
  --utt2spk data/baseline/vox2_train_voxca/utt2spk \
  --age-label-file /xmudata/pzj/vox-ca/vox2dev/segment2age.npy \
  --raw-embedding-scp exp/acsm_raw/embeddings.scp \
  --canonical-embedding-scp exp/acsm_canonical/embeddings.scp \
  --output-json exp/acsm_trajectory_diag.json
```

Model mode:

```bash
python tools/diagnose_acsm_trajectory.py \
  --mode model \
  --config examples/voxceleb/v2/conf/resnet34_acsm_main.yaml \
  --checkpoint exp/ACSM-ResNet34-main-TSTP-emb256-fbank80/models/final_model.pt \
  --wav-scp data/baseline/vox1/wav.scp \
  --utt2spk data/baseline/vox1/utt2spk \
  --age-label-file /xmudata/pzj/vox-ca/vox2dev/segment2age.npy \
  --max-utts 2000 \
  --max-pairs 200000 \
  --batch-size 32 \
  --device cuda \
  --save-utterance-diagnostics exp/acsm_trajectory/utt_diag.jsonl \
  --save-embeddings-dir exp/acsm_trajectory/embeddings \
  --output-json exp/acsm_trajectory/diagnosis.json
```

Embedding mode is for already extracted raw/canonical embeddings. If
`--diagnostic-json` is not provided, internal ACSM state such as gate,
residual, uncertainty, and age posterior entropy is unavailable and explicitly
listed in `unavailable_metrics`.

Model mode loads `config + checkpoint`, reuses the project Dataset/fbank path,
forwards each utterance through ACSM, and reads `raw_embedding`, `embedding`,
`age_posterior`, `gate`, `uncertainty`, and `canonical_residual` directly from
the model output. It can therefore generate a complete diagnosis JSON without
depending on an external `diagnose_acsm.py` result.

The script reports:

- `same_speaker_cross_age_distance_raw`
- `same_speaker_cross_age_distance_canonical`
- `same_speaker_cross_age_delta`
- `different_speaker_distance_raw`
- `different_speaker_distance_canonical`
- `different_speaker_distance_delta`
- `same_speaker_same_age_distance_raw`
- `same_speaker_same_age_distance_canonical`
- `path_valid_pair_count`
- `path_nonzero_batch_ratio`
- `gate_mean`
- `residual_norm_mean`
- `raw_can_cosine_mean`
- `age_gap_bucket_metrics`
- `oracle_age_used`
- `unavailable_metrics`
- `warnings`

Distance is cosine distance: `1 - cosine_similarity`. Delta is
`canonical_mean - raw_mean`, so a negative same-speaker cross-age delta means
canonicalization reduced that distance.

`gate_mean/gate_std/gate_min/gate_max` summarize the scalar canonicalization
gate. Values near zero suggest the canonicalizer may be inactive.

`residual_norm_mean/residual_norm_std` summarize the canonical residual
magnitude. Very small values with `raw_can_cosine_mean` near 1 suggest a
near-identity mapping.

`uncertainty_mean` and `age_posterior_entropy_mean` summarize the predicted age
posterior confidence. These come from the model prediction, not from test age
labels.

`unavailable_metrics` explains every metric that is `null`; null values should
not be silently ignored.

For fair evaluation, model mode never passes true `age_group` to model forward.
The JSON report always contains `"oracle_age_used": false`. The age label file
is used only after inference for pair grouping and age-gap diagnostics.

## Interpretation

If canonicalization clearly lowers same-speaker cross-age distance and
different-speaker distance does not clearly decrease, this supports the ACSM
trajectory claim.

If both same-speaker and different-speaker distances decrease, that may indicate
global contraction or embedding collapse rather than useful age canonicalization.

If same-speaker cross-age distance is nearly unchanged, the canonicalizer may be
near identity.

If `lambda_path > 0` does not improve same-speaker cross-age distance over
`lambda_path = 0`, path consistency should not be emphasized as learning an
effective age trajectory.

This diagnostic supports claim analysis but is not a substitute for EER,
minDCF, ordinary SV evaluation, or multi-seed significance checks.

## Active ACSM Config Set

The active branch retains these ACSM configs:

- `examples/voxceleb/v2/conf/resnet34_acsm.yaml`
- `examples/voxceleb/v2/conf/resnet34_acsm_main.yaml`
- `examples/voxceleb/v2/conf/resnet34_acsm_main_v3.yaml`

Only real experiments plus effectiveness diagnostics can justify treating path
consistency or canonicalization as a useful mechanism.

## Parameter-Matched Control

Model entry:

```python
get_speaker_model("ResNet34_ParamMatch")
```

Config:

```bash
examples/voxceleb/v2/conf/resnet34_parammatch.yaml
```

`ResNet34_ParamMatch` is based on official WeSpeaker ResNet34. It adds a
non-age residual MLP and gate after the embedding layer. The conditioning signal
is input-derived embedding content plus a learnable token. It does not use:

- age labels;
- age posterior;
- `Stage2AgeObserver`;
- `AgeFiLM2d`;
- ordered age canonicalization;
- age loss;
- path loss.

It returns a normal ResNet-style speaker embedding tuple and trains with the
ordinary speaker loss path. This control helps test whether ACSM gains, if any,
come from age-aware canonicalization rather than merely from extra parameters.

## Profile

Command:

```bash
python tools/profile_model.py \
  --config examples/voxceleb/v2/conf/resnet34_acsm_main.yaml \
  --include-baseline \
  --include-parammatch \
  --device cpu \
  --batch-size 1 \
  --frames 80 \
  --output-json exp/model_profile.json
```

The profile reports total parameters, trainable parameters, extra parameters
over ResNet34, ACSM-specific parameters, ParamMatch-specific parameters,
forward latency, input shape, and embedding dimension. FLOPs/MACs are not
computed because no extra dependency is introduced.

## Current Status

The tools and configs validate the experimental plumbing. They do not establish
that ACSM improves EER, reduces minDCF, preserves ordinary SV, or learns a true
age trajectory. Those claims require real training, ordinary and cross-age
evaluation, `lambda_path` ablation, diagnostics, and multi-seed analysis.
