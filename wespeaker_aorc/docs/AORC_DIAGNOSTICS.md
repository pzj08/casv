# AORC Diagnostics

This branch treats AORC as ordinal age modeling plus age proxy learning,
speaker-conditioned age-direction regularization, optional age residual
compensation, and optional cross-age aggregation diagnostics.

Recommended default:

- Use OAM + ORC (`baseline_resnet34_aorc_oam_orc.yaml` or the recommended
  `baseline_resnet34_aorc_full.yaml`).
- Use CAA-X and CAA-LG only as experimental regularizers.
- Keep legacy CAA only for reproduction (`baseline_resnet34_aorc_full_legacy.yaml`).

CAA is provided as an experimental regularizer. It should be enabled only when
batch diagnostics confirm sufficient same-speaker cross-age positives and
validation shows gains, especially on large-age-gap trials.

Run batch diagnostics before training CAA variants:

```bash
python examples/voxceleb/v2/diagnose_aorc_batches.py \
  --config examples/voxceleb/v2/conf/baseline_resnet34_aorc_caa_x.yaml \
  --data_type shard \
  --train_data data/baseline/vox2_train_voxca/shard.list \
  --train_label data/baseline/vox2_train_voxca/utt2spk
```

Key fields:

- `fraction_batches_with_caa_positive`
- `fraction_batches_with_dir_positive`
- `same_speaker_cross_age_pairs`
- `same_speaker_large_gap_pairs`
- `age_group_histogram`

Report CAA experiments with age-gap bucketed EER/minDCF. A zero direction or
CAA loss on many batches usually means the batch sampler did not produce valid
same-speaker cross-age pairs, not that the regularizer is effective.
