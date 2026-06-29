# Environment Manifest

Baseline environment source:

- `requirements.txt`
- PyTorch and torchaudio versions compatible with the local CUDA runtime

Additional baseline dependencies:

- `torchnet==0.0.4`: required by `wespeaker/utils/executor.py`.
- `visdom==0.2.4`: required by `torchnet` top-level imports.
- `tornado==6.5.7`, `jsonpatch==1.33`, `websocket-client==1.9.0`:
  pulled in for the `visdom` import path.

The clean baseline tree does not require method-specific packages.
