# ACSM Claim Guidelines

## Avoid These Claims Without Strong Evidence

- "Solves cross-age speaker verification."
- "Completely removes age effects."
- "Learns the true biological voice-aging trajectory."
- "Does not harm ordinary ASV."
- "Fair evaluation is fully guaranteed."
- "Significantly outperforms all baselines."

## Prefer These Claims

- "Mitigates age-related within-speaker shift in cross-age positive trials."
- "Learns an ordered age-conditioned canonical transformation."
- "Default evaluation uses predicted age posterior and does not use true test
  age."
- "Consistency loss and gate constraints reduce the risk of ordinary-SV
  degradation."
- "ACSM is evaluated under the same training and Vox-CA/MIM evaluation
  protocol as the baselines."

## Required Evidence

- Cross-age EER/minDCF improvements require matched training data, matched
  trial lists, and no test-age oracle path.
- "Age trajectory" claims require `lambda_path > 0` ablation and diagnostics
  showing nontrivial path/residual behavior.
- "No ordinary ASV degradation" requires Vox-O/E/H results. If ordinary SV
  drops more than 5% relative, the main conclusion must be weakened.
- If EER improvement is below 3% relative, report multi-seed results and
  statistical significance analysis before using "significant".
- Oracle age diagnostics must be labeled as oracle and cannot be used as the
  main result.

## Current Status

Current ACSM code has unit and smoke tests only. It has no real EER evidence
and no supported claim of performance improvement yet.
