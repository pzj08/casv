# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang)
#               2021 Hongji Wang (jijijiang77@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import logging


def _is_acsm_key(key):
    return (key.startswith('age_observer.') or key.startswith('age_film')
            or key.startswith('canonicalizer.') or key.startswith('path_loss.'))


def load_checkpoint(model: torch.nn.Module,
                    path: str,
                    strict: bool = False,
                    allow_acsm_partial: bool = False,
                    logger=None):
    """
    Load a checkpoint and handle potential size mismatch in
    the projection layer.
    """
    log = logger or logging
    checkpoint = torch.load(path, map_location="cpu")
    checkpoint = checkpoint['model'] if isinstance(
        checkpoint, dict) and 'model' in checkpoint else checkpoint
    current_state_dict = model.state_dict()

    proj_key = "projection.weight"
    if proj_key in checkpoint and proj_key in current_state_dict:
        ckpt_w = checkpoint[proj_key]
        curr_w = current_state_dict[proj_key]

        # Check if shapes mismatch
        if ckpt_w.shape != curr_w.shape:
            log.warning(
                f"Size mismatch for {proj_key}: "
                f"checkpoint has shape {ckpt_w.shape}, "
                f"current model has shape {curr_w.shape}."
            )

            ckpt_len = ckpt_w.shape[0]
            curr_len = curr_w.shape[0]

            # Case: checkpoint from speed-perturbed training
            # (num_classes * 3) to LMFT training (original num_classes)
            if ckpt_len > curr_len:
                log.info(
                    "Loading the first %d rows from checkpoint's "
                    "projection layer.",
                    curr_len,
                )
                # Only use the first part of weights from checkpoint
                checkpoint[proj_key] = ckpt_w[:curr_len, :]

                # Also handle bias if present
                bias_key = "projection.bias"
                if bias_key in checkpoint and bias_key in current_state_dict:
                    ckpt_b = checkpoint[bias_key]
                    if ckpt_b.shape[0] > curr_len:
                        checkpoint[bias_key] = ckpt_b[:curr_len]

    if strict:
        model.load_state_dict(checkpoint, strict=True)
        log.info("checkpoint loaded strictly: %s", path)
        return {
            'loaded_key_count': len(checkpoint),
            'missing_keys': [],
            'unexpected_keys': [],
            'missing_acsm_key_count': 0,
        }

    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint,
        strict=False,
    )
    loaded_key_count = sum(1 for k in checkpoint
                           if k in current_state_dict
                           and checkpoint[k].shape == current_state_dict[k].shape)

    # Filter out projection keys we already handled explicitly so logs
    # focus on truly unexpected tensors.
    final_unexpected_keys = [
        k for k in unexpected_keys if "projection" not in k
    ]

    missing_acsm = [k for k in missing_keys if _is_acsm_key(k)]
    non_acsm_missing = [
        k for k in missing_keys if not _is_acsm_key(k) and "projection" not in k
    ]

    if allow_acsm_partial and (non_acsm_missing or final_unexpected_keys):
        raise RuntimeError(
            "ACSM partial checkpoint load found non-ACSM mismatch: "
            "missing={}, unexpected={}".format(non_acsm_missing[:20],
                                               final_unexpected_keys[:20]))

    for key in missing_keys:
        # Missing projection keys are expected if the source model did
        # not have projection; do not warn for those.
        if "projection" not in key:
            log.warning("missing tensor: %s", key)

    for key in final_unexpected_keys:
        log.warning("unexpected tensor: %s", key)

    report = {
        'loaded_key_count': loaded_key_count,
        'missing_key_count': len(missing_keys),
        'unexpected_key_count': len(unexpected_keys),
        'missing_acsm_key_count': len(missing_acsm),
        'missing_keys': missing_keys,
        'unexpected_keys': unexpected_keys,
        'unexpected_key_examples': final_unexpected_keys[:20],
    }
    log.info("checkpoint load report: %s", report)
    return report


def save_checkpoint(model: torch.nn.Module, path: str):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, path)
