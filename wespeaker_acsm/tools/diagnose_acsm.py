#!/usr/bin/env python3

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import yaml

from wespeaker.dataset.dataset import Dataset
from wespeaker.dataset.dataset_utils import apply_cmvn
from wespeaker.models.acsm_modules import get_acsm_config
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.checkpoint import load_checkpoint


def _entropy(q, eps=1.0e-12):
    return -(q.clamp_min(eps) * q.clamp_min(eps).log()).sum(dim=-1)


def _stats_by_group(values, groups, num_groups, ignore_index):
    out = {}
    for group in range(num_groups):
        mask = groups == group
        out[str(group)] = (float(values[mask].mean().item())
                           if mask.any() else None)
    out['ignore'] = (float(values[groups == ignore_index].mean().item())
                     if (groups == ignore_index).any() else None)
    return out


def _load_batch(configs, args, device):
    if args.fake:
        feats = torch.randn(args.batch_size, args.num_frames,
                            configs['model_args']['feat_dim'])
        age_group = torch.full((args.batch_size, ), -1, dtype=torch.long)
        return feats.to(device), age_group.to(device)

    dataset_conf = dict(configs['dataset_args'])
    dataset_conf['shuffle'] = False
    dataset_conf['filter'] = False
    dataset_conf['spec_aug'] = False
    dataset_conf['speed_perturb'] = False
    dataset = Dataset(args.data_type,
                      args.data_list,
                      dataset_conf,
                      spk2id_dict={},
                      whole_utt=False,
                      repeat_dataset=False)
    rows = []
    for sample in dataset:
        rows.append(sample)
        if len(rows) >= args.batch_size:
            break
    if not rows:
        raise RuntimeError('no samples loaded for ACSM diagnosis')
    feats = torch.stack([row['feat'] for row in rows]).float()
    feats = apply_cmvn(feats, **dataset_conf.get('cmvn_args', {}))
    age_group = torch.full((len(rows), ), -1, dtype=torch.long)
    return feats.to(device), age_group.to(device)


def main():
    parser = argparse.ArgumentParser(
        description='Diagnose whether ACSM canonicalization is active.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-list')
    parser.add_argument('--data-type', default='raw')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-frames', type=int, default=200)
    parser.add_argument('--fake', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output-json')
    args = parser.parse_args()

    with open(args.config) as f:
        configs = yaml.safe_load(f)
    configs.setdefault('model_args', {})
    configs['model_args']['acsm_args'] = get_acsm_config(configs)
    if not args.fake and not args.data_list:
        raise ValueError('--data-list is required unless --fake is set')

    device = torch.device(args.device if (
        args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    model = get_speaker_model(configs['model'])(**configs['model_args'])
    load_checkpoint(model, args.checkpoint, strict=False)
    model.to(device).eval()
    feats, age_group = _load_batch(configs, args, device)

    with torch.no_grad():
        outputs = model(feats)

    e_raw = F.normalize(outputs['raw_embedding'], dim=-1, eps=1.0e-12)
    e_can = F.normalize(outputs['embedding'], dim=-1, eps=1.0e-12)
    q_age = outputs['age_posterior']
    residual_norm = outputs['canonical_residual'].norm(dim=-1)
    gate = outputs['gate'].view(-1)
    transitions = model.canonicalizer.adjacent_transitions.detach()
    paths = model.canonicalizer._paths_to_reference().detach()

    report = {
        'gate_mean': float(gate.mean().item()),
        'gate_std': float(gate.std(unbiased=False).item()),
        'gate_min': float(gate.min().item()),
        'gate_max': float(gate.max().item()),
        'uncertainty_mean': float(outputs['uncertainty'].mean().item()),
        'age_posterior_entropy_mean': float(_entropy(q_age).mean().item()),
        'residual_norm_mean': float(residual_norm.mean().item()),
        'residual_norm_std': float(residual_norm.std(unbiased=False).item()),
        'raw_can_cosine_mean': float((e_raw * e_can).sum(dim=-1).mean().item()),
        'raw_can_l2_mean': float((e_raw - e_can).norm(dim=-1).mean().item()),
        'transition_norms': [float(x) for x in transitions.norm(dim=-1).cpu()],
        'path_norm_by_age_group': [
            float(x) for x in paths.norm(dim=-1).cpu()
        ],
        'gate_mean_by_age_group': _stats_by_group(
            gate, age_group, model.num_age_groups, model.ignore_age_index),
        'residual_norm_mean_by_age_group': _stats_by_group(
            residual_norm, age_group, model.num_age_groups,
            model.ignore_age_index),
        'notes': [
            'gate_mean near 0 suggests ACSM may be inactive.',
            'raw_can_cosine_mean near 1 with small residual_norm suggests near identity.',
            'large residual_norm with ordinary SV degradation suggests over-compensation.',
        ],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)),
                    exist_ok=True)
        with open(args.output_json, 'w') as f:
            f.write(text + '\n')


if __name__ == '__main__':
    main()
