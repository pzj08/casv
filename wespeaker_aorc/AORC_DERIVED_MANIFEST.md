# AORC Derived Manifest

This tree is an innovation branch derived from `wespeaker_baseline`.
It is not the authoritative baseline tree.

## Baseline Inheritance

`examples/voxceleb/v2/conf/baseline_resnet34_aorc_off.yaml` mirrors the
baseline training parameters and only appends disabled `aorc_args`.
Formal baseline numbers must still come from `wespeaker_baseline`.

## Added AORC Files

- `wespeaker/losses/aorc_losses.py`
- `wespeaker/models/aorc_modules.py`
- `tests/test_aorc.py`

## Modified AORC Integration Points

- `wespeaker/bin/train.py`
- `wespeaker/bin/extract.py`
- `wespeaker/dataset/dataset.py`
- `wespeaker/dataset/processor.py`
- `wespeaker/utils/executor.py`

## Recipe Files

- `examples/voxceleb/v2/conf/baseline_resnet34_aorc_off.yaml`
- `examples/voxceleb/v2/conf/baseline_resnet34_oam.yaml`
- `examples/voxceleb/v2/conf/baseline_resnet34_oam_orc.yaml`
- `examples/voxceleb/v2/conf/baseline_resnet34_oam_caa.yaml`
- `examples/voxceleb/v2/conf/baseline_resnet34_aorc_full.yaml`
- `examples/voxceleb/v2/run_aorc_baseline_ablation.sh`

Run `aorc_off` as a smoke guard. Its logs should not include AORC losses.
Run enabled variants to confirm age labels load and AORC losses are non-zero.
