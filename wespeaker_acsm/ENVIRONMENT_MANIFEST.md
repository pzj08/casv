# Environment Manifest

Base environment source:

- `requirements.txt`
- PyTorch and torchaudio versions compatible with the local CUDA runtime

Additional ACSM package dependencies:

- No ACSM-specific packages.
- Inherits the baseline runtime additions:
  `torchnet==0.0.4`, `visdom==0.2.4`, `tornado==6.5.7`,
  `jsonpatch==1.33`, `websocket-client==1.9.0`.

ACSM uses PyTorch modules only. The age-label loader accepts text mappings or
NumPy `.npy` dictionaries through dependencies already used by WeSpeaker.
