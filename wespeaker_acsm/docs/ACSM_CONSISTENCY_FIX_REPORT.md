# ACSM Consistency Loss Fix Report

## Summary

The previous ACSM-main run used unnormalized raw-L2 consistency:

```python
(e_can - e_raw.detach()).pow(2).sum(dim=-1).mean()
```

Early training showed that this term dominated the objective after weighting.
The old raw-L2 run was removed from the active experiment directory and must
not be reported as ACSM-main.

## Fixed Behavior

`consistency.type: cosine` is now the main setting. It computes:

```python
1 - cosine_similarity(normalize(e_can), normalize(e_raw.detach()))
```

This matches cosine-based speaker verification better and removes the raw
embedding norm and dimension-sum scale factor from the ordinary-SV protection
term.

`consistency.type: raw_l2_sum` is not part of the active v2/v3 settings.

## Updated Config Policy

- ACSM v2/v3 configs use `consistency.type: cosine`.
- Raw-L2 legacy configs have been removed from the active branch.

## Smoke-Rerun Acceptance Criteria

For the first 5-10 epochs of the cosine ACSM-main rerun:

- `weighted_consistency` should not dominate `loss_spk`.
- `loss_spk` should decrease and accuracy should rise.
- `loss_age`, `loss_path`, and `loss_smooth` should remain finite.
- `gate_mean` should rise gradually.
- `raw_can_cosine_mean` should not remain exactly `1.000000` for the full run.
- `path_nonzero_batch_ratio` should remain nonzero.

This fix does not establish any EER or minDCF improvement.
