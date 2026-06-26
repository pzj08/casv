#!/usr/bin/env python3

import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from wespeaker.models.acsm_modules import acsm_is_enabled, get_acsm_config
from wespeaker.models.aorc_modules import AORCWrapper, aorc_is_enabled, get_aorc_config
from wespeaker.models.speaker_model import get_speaker_model


def _build(configs):
    configs = copy.deepcopy(configs)
    if acsm_is_enabled(configs):
        configs['model_args']['acsm_args'] = get_acsm_config(configs)
    model = get_speaker_model(configs['model'])(**configs['model_args'])
    if aorc_is_enabled(configs):
        model = AORCWrapper(model, configs['model_args']['embed_dim'],
                            get_aorc_config(configs))
    return model


def _param_stats(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    acsm_extra = sum(p.numel() for name, p in model.named_parameters()
                     if name.startswith(('age_observer', 'age_film',
                                         'canonicalizer')))
    parammatch_extra = sum(p.numel() for name, p in model.named_parameters()
                           if name.startswith('param_match'))
    return total, trainable, acsm_extra, parammatch_extra


def _latency(model, device, feat_dim, frames, batch_size, warmup, iters):
    model.to(device).eval()
    x = torch.randn(batch_size, frames, feat_dim, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(iters):
            model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / float(iters)


def _profile(name, configs, args):
    model = _build(configs)
    total, trainable, acsm_extra, parammatch_extra = _param_stats(model)
    feat_dim = configs['model_args']['feat_dim']
    device = torch.device(args.device if (
        args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    latency = _latency(model, device, feat_dim, args.frames, args.batch_size,
                       args.warmup, args.iters)
    return {
        'name': name,
        'model': configs['model'],
        'total_params': total,
        'trainable_params': trainable,
        'acsm_extra_params': acsm_extra,
        'parammatch_extra_params': parammatch_extra,
        'extra_params_over_resnet34': None,
        'flops_or_macs': None,
        'flops_note': 'FLOPs/MACs not computed; no extra dependency is used.',
        'device': str(device),
        'forward_latency_ms': latency,
        'batch_size': args.batch_size,
        'input_frames': args.frames,
        'feat_dim': feat_dim,
        'embedding_dim': configs['model_args']['embed_dim'],
    }


def main():
    parser = argparse.ArgumentParser(
        description='Profile speaker model params and fake-input latency.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--include-baseline', action='store_true')
    parser.add_argument('--include-parammatch', action='store_true')
    parser.add_argument('--include-aorc', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--frames', type=int, default=200)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--iters', type=int, default=5)
    parser.add_argument('--output-json')
    args = parser.parse_args()

    with open(args.config) as f:
        base = yaml.safe_load(f)
    baseline_cfg = copy.deepcopy(base)
    baseline_cfg['model'] = 'ResNet34'
    baseline_cfg['model_args'].pop('acsm_args', None)
    baseline_cfg['model_args'].pop('param_match_args', None)
    baseline_total = _param_stats(_build(baseline_cfg))[0]

    reports = [_profile('configured', base, args)]
    if args.include_baseline:
        reports.append(_profile('resnet34_baseline', baseline_cfg, args))
    if args.include_parammatch:
        cfg = copy.deepcopy(base)
        cfg['model'] = 'ResNet34_ParamMatch'
        cfg['model_args'].pop('acsm_args', None)
        cfg['model_args'].setdefault('param_match_args', {
            'bottleneck_dim': 64,
            'residual_scale': 0.1,
        })
        reports.append(_profile('resnet34_parammatch', cfg, args))
    if args.include_aorc:
        cfg = copy.deepcopy(base)
        cfg['model'] = 'ResNet34'
        cfg['model_args'].pop('acsm_args', None)
        cfg['model_args'].pop('param_match_args', None)
        cfg['aorc_args'] = {
            'enable_oam': True,
            'enable_orc': True,
            'enable_caa': False,
            'num_age_groups': 7,
        }
        reports.append(_profile('resnet34_aorc_oam_orc', cfg, args))
    for report in reports:
        report['extra_params_over_resnet34'] = (
            report['total_params'] - baseline_total)
    text = json.dumps({'profiles': reports}, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        with open(args.output_json, 'w') as f:
            f.write(text + '\n')


if __name__ == '__main__':
    main()
