# ACSM Effectiveness Diagnostics

This diagnostic supports mechanistic analysis of ACSM but does not replace official EER/minDCF evaluation.

## Why This Exists

Non-zero `gate`, `residual_norm`, or `loss_path` only proves that the ACSM code path is active. It does not prove that canonicalization helps speaker verification, learns an age trajectory, or avoids damaging speaker separation.

The effectiveness diagnostic asks three narrower questions:

1. Does ACSM actually change the embedding?
2. Does canonicalization reduce same-speaker cross-age distance?
3. Does canonicalization avoid compressing different-speaker distance?

These are pair-distance diagnostics, not final SV metrics.

## Metrics

### Embedding Change

The report computes:

```text
raw_can_cosine_mean/std/min/p05/p50/p95
delta_raw_l2_mean/std
delta_norm_l2_mean/std
identity_like
```

`delta_raw_l2` is `||e_can - e_raw||`. `delta_norm_l2` is `||normalize(e_can) - normalize(e_raw)||`.

If `raw_can_cosine_mean` is near `1.000000` and `delta_norm_l2_mean` is near zero, ACSM is near-identity for the evaluated samples.

### Same-Speaker Cross-Age

Pairs are selected where `spk_i == spk_j` and `age_group_i != age_group_j`.

```text
distance = 1 - cosine_similarity
delta_mean_raw_minus_can = raw_distance_mean - canonical_distance_mean
```

Positive delta means canonicalization moved same-speaker cross-age samples closer.

### Different-Speaker

Pairs are selected where `spk_i != spk_j` and randomly sampled with a fixed seed.

If same-speaker cross-age distance drops but different-speaker distance also drops substantially, the result may be global compression or collapse rather than useful age alignment.

### Age Gap Buckets

For continuous age labels, same-speaker pairs are also summarized for:

```text
gap >= 5
gap >= 10
gap >= 15
gap >= 20
```

Each bucket reports pair count, raw distance, canonical distance, delta, and improved-pair ratio. Buckets with no pairs are marked unreliable.

## Decision Rules

Defaults:

```text
embedding_changed:
  raw_can_cosine_mean < 0.9999 or delta_norm_l2_mean > 1e-5

same_cross_age_improved:
  same_cross_age_delta_mean > 0 and improved_pair_ratio > 0.5

different_speaker_preserved:
  diff_speaker_delta_relative < 0.02

collapse_risk:
  diff_speaker_delta_relative >= 0.05

near_identity_risk:
  raw_can_cosine_mean >= 0.9999 and delta_norm_l2_mean <= 1e-5
```

Overall:

```text
PASS:
  embedding_changed=True
  same_cross_age_improved=True
  different_speaker_preserved=True
  collapse_risk=False

PARTIAL:
  same_cross_age_improved=True, but embedding change is weak or
  different-speaker preservation is uncertain

FAIL:
  near_identity_risk=True, or same_cross_age_improved=False, or
  collapse_risk=True
```

These rules are conservative diagnostic gates. They are not EER/minDCF claims.

## Bootstrap

Use `--bootstrap 1000` to add 95% bootstrap confidence intervals for:

```text
same_cross_age_delta_mean_ci95
diff_speaker_delta_mean_ci95
same_cross_age_improved_pair_ratio_ci95
```

Bootstrap here only describes pair-distance diagnostic stability.

## Embedding Mode Example

```bash
python tools/diagnose_acsm_trajectory.py \
  --mode embedding \
  --raw-embeddings exp/acsm/raw.npy \
  --canonical-embeddings exp/acsm/canonical.npy \
  --utt-list exp/acsm/utts.txt \
  --utt2spk data/eval/utt2spk \
  --age-label-file data/eval/age_labels \
  --effectiveness-report \
  --output-json exp/acsm/acsm_effectiveness.json \
  --max-same-pairs 200000 \
  --max-diff-pairs 200000 \
  --seed 1234
```

Kaldi scp input remains supported through `--raw-embedding-scp` and `--canonical-embedding-scp`.

## Model Mode Example

```bash
python tools/diagnose_acsm_trajectory.py \
  --mode model \
  --config examples/voxceleb/v2/conf/resnet34_acsm_main.yaml \
  --checkpoint exp/acsm/model_10.pt \
  --data-list data/eval/feats.scp \
  --data-type feat \
  --utt2spk data/eval/utt2spk \
  --age-label-file data/eval/age_labels \
  --effectiveness-report \
  --output-json exp/acsm/acsm_effectiveness_model.json \
  --max-utts 5000 \
  --max-same-pairs 200000 \
  --max-diff-pairs 200000 \
  --seed 1234
```

Model mode sets `"oracle_age_used": false`. Age labels are used only for diagnostic grouping and are not passed into model forward to change outputs.

## Example JSON Shape

```json
{
  "embedding_change": {
    "raw_can_cosine_mean": 0.99995,
    "delta_norm_l2_mean": 0.00001,
    "identity_like": true
  },
  "same_speaker_cross_age": {
    "raw_distance_mean": 0.22,
    "canonical_distance_mean": 0.18,
    "delta_mean_raw_minus_can": 0.04,
    "improved_pair_ratio": 0.62,
    "num_pairs": 1200
  },
  "different_speaker": {
    "raw_distance_mean": 0.91,
    "canonical_distance_mean": 0.90,
    "delta_mean_raw_minus_can": 0.01,
    "delta_relative": 0.011,
    "bad_compress_ratio": 0.03,
    "num_pairs": 200000
  },
  "effectiveness_decision": {
    "embedding_changed": true,
    "same_cross_age_improved": true,
    "different_speaker_preserved": true,
    "collapse_risk": false,
    "near_identity_risk": false,
    "overall": "PASS"
  }
}
```

## Claim Limits

Allowed:

```text
ACSM changed embeddings under the diagnostic sample.
ACSM reduced same-speaker cross-age pair distance in this diagnostic.
Different-speaker distance was preserved under the configured threshold.
```

Not allowed:

```text
ACSM improves EER.
ACSM outperforms baseline.
ACSM learns true age trajectory.
ACSM does not hurt speaker verification.
```

Those claims require official EER/minDCF evaluation and controlled ablations.
