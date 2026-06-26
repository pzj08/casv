#!/usr/bin/env python3

import argparse
import copy
import json
import os
import random
import re
import sys
import tempfile
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kaldiio
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from wespeaker.dataset.dataset import Dataset
from wespeaker.dataset.dataset_utils import apply_cmvn, spec_aug
from wespeaker.models.acsm_modules import get_acsm_config
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.checkpoint import load_checkpoint


def _read_utt2spk(path):
    out = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                out[parts[0]] = parts[1]
    return out


def _age_value_to_group(value, bins):
    for idx, upper in enumerate(bins):
        if value < upper:
            return idx
    return len(bins)


def _key_candidates(key):
    base = key[:-4] if key.endswith('.wav') else key
    items = [key, base, base.replace('/', '-'), base.replace('-', '/')]
    parts = base.replace('-', '/').split('/')
    if len(parts) >= 2:
        items.extend(['/'.join(parts[:2]), '-'.join(parts[:2])])
    out = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


def _read_age_labels(path, keys, label_type, bins, ignore_index):
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.ndarray) and loaded.shape == ():
        loaded = loaded.item()
    if not isinstance(loaded, dict):
        raise ValueError('age_label_file must be a dict-like npy file')
    ages = {}
    for key in keys:
        value = None
        for cand in _key_candidates(key):
            if cand in loaded:
                value = loaded[cand]
                break
        if value is None:
            ages[key] = ignore_index
            continue
        if isinstance(value, (list, tuple, np.ndarray)):
            value = value[0]
        value = float(value)
        ages[key] = (_age_value_to_group(value, bins)
                     if label_type == 'value' else int(value))
    return ages


def _read_embeddings(path, max_utts=None):
    out = {}
    for key, emb in kaldiio.load_scp_sequential(path):
        out[key] = np.asarray(emb, dtype=np.float32)
        if max_utts and len(out) >= max_utts:
            break
    return out


def _cosine_distance(a, b, eps=1.0e-12):
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), eps)
    return 1.0 - float(np.dot(a, b) / denom)


def _parse_trial_pairs(path):
    pairs = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            if parts[0] in ('target', 'nontarget', '1', '0'):
                pairs.append((parts[1], parts[2]))
            else:
                pairs.append((parts[0], parts[1]))
    return pairs


def _add_pair(buckets, a, b, utt2spk, ages, ignore_index, max_pairs):
    if a not in utt2spk or b not in utt2spk:
        return
    age_a = ages.get(a, ignore_index)
    age_b = ages.get(b, ignore_index)
    same = utt2spk[a] == utt2spk[b]
    if same and age_a != ignore_index and age_b != ignore_index:
        name = 'same_same_age' if age_a == age_b else 'same_cross'
    elif not same:
        name = 'different'
    else:
        return
    if max_pairs is None or len(buckets[name]) < max_pairs:
        buckets[name].append((a, b))


def _sample_pairs(keys, utt2spk, ages, ignore_index, max_pairs, trial_file):
    rng = random.Random(3407)
    keys = list(keys)
    key_set = set(keys)
    buckets = {'same_cross': [], 'same_same_age': [], 'different': []}
    if trial_file:
        for a, b in _parse_trial_pairs(trial_file):
            if a in key_set and b in key_set:
                _add_pair(buckets, a, b, utt2spk, ages, ignore_index,
                          max_pairs)
        return buckets

    by_spk = defaultdict(list)
    for key in keys:
        if key in utt2spk:
            by_spk[utt2spk[key]].append(key)
    for utts in by_spk.values():
        rng.shuffle(utts)
        for i in range(len(utts)):
            for j in range(i + 1, len(utts)):
                _add_pair(buckets, utts[i], utts[j], utt2spk, ages,
                          ignore_index, max_pairs)

    if len(keys) >= 2:
        attempts = 0
        target = max_pairs or min(200000, len(keys) * 10)
        while len(buckets['different']) < target and attempts < target * 50:
            a, b = rng.sample(keys, 2)
            attempts += 1
            if utt2spk.get(a) != utt2spk.get(b):
                _add_pair(buckets, a, b, utt2spk, ages, ignore_index,
                          max_pairs)
    return buckets


def _pair_stats(pairs, raw_embs, can_embs):
    raw, can = [], []
    for a, b in pairs:
        raw.append(_cosine_distance(raw_embs[a], raw_embs[b]))
        can.append(_cosine_distance(can_embs[a], can_embs[b]))
    if not raw:
        return {'count': 0, 'raw_mean': None, 'canonical_mean': None,
                'delta': None}
    raw_mean = float(np.mean(raw))
    can_mean = float(np.mean(can))
    return {
        'count': len(raw),
        'raw_mean': raw_mean,
        'canonical_mean': can_mean,
        'delta': can_mean - raw_mean,
    }


def _age_gap_metrics(pairs, raw_embs, can_embs, ages):
    buckets = defaultdict(lambda: {'raw': [], 'canonical': []})
    for a, b in pairs:
        gap = abs(int(ages[a]) - int(ages[b]))
        buckets[str(gap)]['raw'].append(_cosine_distance(raw_embs[a],
                                                         raw_embs[b]))
        buckets[str(gap)]['canonical'].append(
            _cosine_distance(can_embs[a], can_embs[b]))
    out = {}
    for gap, values in sorted(buckets.items(), key=lambda x: int(x[0])):
        raw_mean = float(np.mean(values['raw']))
        can_mean = float(np.mean(values['canonical']))
        out[gap] = {
            'count': len(values['raw']),
            'raw_mean': raw_mean,
            'canonical_mean': can_mean,
            'delta': can_mean - raw_mean,
        }
    return out


def _read_path_ratio(train_log):
    if not train_log:
        return None
    values = []
    pattern = re.compile(r'path_valid_pair_count=([0-9.]+)')
    ratio_pattern = re.compile(r'path_nonzero_batch_ratio=([0-9.]+)')
    last_ratio = None
    with open(train_log, 'r', errors='ignore') as f:
        for line in f:
            ratio_match = ratio_pattern.search(line)
            if ratio_match:
                last_ratio = float(ratio_match.group(1))
            for match in pattern.finditer(line):
                values.append(float(match.group(1)))
    if last_ratio is not None:
        return last_ratio
    if values:
        return float(sum(v > 0.0 for v in values) / len(values))
    return None


def _read_diagnostic_json(path):
    if not path:
        return {}
    with open(path, 'r') as f:
        return json.load(f)


def _mean_std(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values))


def _entropy(q, eps=1.0e-12):
    q = np.asarray(q, dtype=np.float64)
    q = np.clip(q, eps, 1.0)
    return float(-(q * np.log(q)).sum())


def _load_wav_scp_as_raw_list(wav_scp, utt2spk, utt_list):
    keep = None
    if utt_list:
        with open(utt_list, 'r') as f:
            keep = {line.strip().split()[0] for line in f if line.strip()}
    rows = []
    with open(wav_scp, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            utt, wav = parts
            if keep is not None and utt not in keep:
                continue
            rows.append({'key': utt, 'spk': utt2spk.get(utt, 'unknown'),
                         'wav': wav})
    handle = tempfile.NamedTemporaryFile('w', suffix='.jsonl', delete=False)
    for row in rows:
        handle.write(json.dumps(row) + '\n')
    handle.close()
    return handle.name


def _pad_collate(batch):
    keys = [x['key'] for x in batch]
    out = {'key': keys}
    if 'feat' in batch[0]:
        feat_dim = batch[0]['feat'].shape[-1]
        max_t = max(x['feat'].shape[0] for x in batch)
        feats = batch[0]['feat'].new_zeros(len(batch), max_t, feat_dim)
        for idx, item in enumerate(batch):
            feats[idx, :item['feat'].shape[0]] = item['feat']
        out['feat'] = feats
    else:
        raise ValueError('model mode currently expects fbank features')
    return out


def _extract_model_mode(args):
    with open(args.config, 'r') as f:
        configs = yaml.safe_load(f)
    configs = copy.deepcopy(configs)
    configs.setdefault('model_args', {})
    configs['model_args']['acsm_args'] = get_acsm_config(configs)
    device = torch.device(args.device if (
        args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    model = get_speaker_model(configs['model'])(**configs['model_args'])
    load_checkpoint(model, args.checkpoint, strict=False)
    model.to(device).eval()

    utt2spk = _read_utt2spk(args.utt2spk)
    data_list = args.data_list
    temp_list = None
    if args.wav_scp:
        temp_list = _load_wav_scp_as_raw_list(args.wav_scp, utt2spk,
                                              args.utt_list)
        data_list = temp_list
    if not data_list:
        raise ValueError('model mode requires --wav-scp or --data-list')

    dataset_conf = copy.deepcopy(configs['dataset_args'])
    dataset_conf['shuffle'] = False
    dataset_conf['filter'] = False
    dataset_conf['speed_perturb'] = False
    dataset_conf['spec_aug'] = False
    dataset_conf['aug_prob'] = 0.0
    if 'fbank_args' in dataset_conf:
        dataset_conf['fbank_args']['dither'] = 0.0
    dataset = Dataset(args.data_type,
                      data_list,
                      dataset_conf,
                      spk2id_dict={},
                      whole_utt=True,
                      repeat_dataset=False)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        num_workers=0,
                        shuffle=False,
                        collate_fn=_pad_collate)

    raw_embs, can_embs, utt_diag = {}, {}, {}
    with torch.no_grad():
        for batch in loader:
            keys = batch['key']
            feats = batch['feat'].float().to(device)
            if dataset_conf.get('cmvn', True):
                feats = apply_cmvn(feats, **dataset_conf.get('cmvn_args', {}))
            if dataset_conf.get('spec_aug', False):
                feats = spec_aug(feats, **dataset_conf['spec_aug_args'])
            outputs = model(feats)
            if not isinstance(outputs, dict):
                raise TypeError('model mode requires ACSM dict output')
            required = [
                'raw_embedding', 'embedding', 'age_posterior', 'gate',
                'uncertainty', 'canonical_residual'
            ]
            missing = [k for k in required if k not in outputs]
            if missing:
                raise KeyError('ACSM output missing keys: {}'.format(missing))

            raw = outputs['raw_embedding'].detach().cpu()
            can = outputs['embedding'].detach().cpu()
            q_age = outputs['age_posterior'].detach().cpu()
            gate = outputs['gate'].detach().cpu().view(len(keys), -1)
            uncertainty = outputs['uncertainty'].detach().cpu().view(len(keys),
                                                                     -1)
            residual = outputs['canonical_residual'].detach().cpu()
            age_pred = outputs.get('age_pred')
            if age_pred is None:
                age_pred = q_age.argmax(dim=1)
            else:
                age_pred = age_pred.detach().cpu().view(-1)
            raw_norm = F.normalize(raw, dim=-1, eps=1.0e-12)
            can_norm = F.normalize(can, dim=-1, eps=1.0e-12)
            raw_can_cos = (raw_norm * can_norm).sum(dim=-1)
            raw_can_l2 = (raw_norm - can_norm).norm(dim=-1)
            for idx, utt in enumerate(keys):
                raw_embs[utt] = raw[idx].numpy().astype(np.float32)
                can_embs[utt] = can[idx].numpy().astype(np.float32)
                utt_diag[utt] = {
                    'age_pred': int(age_pred[idx].item()),
                    'age_entropy': _entropy(q_age[idx].numpy()),
                    'gate_mean': float(gate[idx].mean().item()),
                    'uncertainty': float(uncertainty[idx].mean().item()),
                    'residual_norm': float(residual[idx].norm().item()),
                    'raw_can_cosine': float(raw_can_cos[idx].item()),
                    'raw_can_l2': float(raw_can_l2[idx].item()),
                }
            if args.max_utts and len(raw_embs) >= args.max_utts:
                keep = list(raw_embs.keys())[:args.max_utts]
                raw_embs = {k: raw_embs[k] for k in keep}
                can_embs = {k: can_embs[k] for k in keep}
                utt_diag = {k: utt_diag[k] for k in keep}
                break
    if temp_list:
        os.unlink(temp_list)
    return raw_embs, can_embs, utt_diag


def _save_utterance_diagnostics(path, keys, utt2spk, ages, utt_diag):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        for utt in keys:
            item = utt_diag.get(utt, {})
            row = {
                'utt': utt,
                'spk': utt2spk.get(utt),
                'age_group': ages.get(utt),
                'age_pred': item.get('age_pred'),
                'age_entropy': item.get('age_entropy'),
                'gate_mean': item.get('gate_mean'),
                'residual_norm': item.get('residual_norm'),
                'raw_can_cosine': item.get('raw_can_cosine'),
            }
            f.write(json.dumps(row, sort_keys=True) + '\n')


def _save_embeddings(path, keys, raw_embs, can_embs, utt_diag):
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, 'raw_embeddings.npy'),
            np.stack([raw_embs[k] for k in keys]).astype(np.float32))
    np.save(os.path.join(path, 'canonical_embeddings.npy'),
            np.stack([can_embs[k] for k in keys]).astype(np.float32))
    with open(os.path.join(path, 'utts.txt'), 'w') as f:
        for utt in keys:
            f.write(utt + '\n')
    with open(os.path.join(path, 'diagnostics.json'), 'w') as f:
        json.dump({utt: utt_diag.get(utt, {}) for utt in keys},
                  f,
                  indent=2,
                  sort_keys=True)


def _build_report(args, raw_embs, can_embs, utt_diag):
    keys = sorted(set(raw_embs) & set(can_embs))
    utt2spk = _read_utt2spk(args.utt2spk)
    bins = [float(x) for x in args.age_bins.split(',') if x]
    ages = _read_age_labels(args.age_label_file, keys, args.age_label_type,
                            bins, args.ignore_age_index)
    valid_keys = [
        k for k in keys
        if k in utt2spk and ages.get(k, args.ignore_age_index) !=
        args.ignore_age_index
    ]
    if args.max_utts:
        valid_keys = valid_keys[:args.max_utts]
    buckets = _sample_pairs(valid_keys, utt2spk, ages, args.ignore_age_index,
                            args.max_pairs, args.trial_file)
    same_cross = _pair_stats(buckets['same_cross'], raw_embs, can_embs)
    same_same = _pair_stats(buckets['same_same_age'], raw_embs, can_embs)
    different = _pair_stats(buckets['different'], raw_embs, can_embs)

    diag_json = _read_diagnostic_json(args.diagnostic_json)
    gate_values = [utt_diag[k].get('gate_mean') for k in valid_keys
                   if k in utt_diag]
    uncertainty_values = [utt_diag[k].get('uncertainty') for k in valid_keys
                          if k in utt_diag]
    entropy_values = [utt_diag[k].get('age_entropy') for k in valid_keys
                      if k in utt_diag]
    residual_values = [utt_diag[k].get('residual_norm') for k in valid_keys
                       if k in utt_diag]
    raw_can_cos_values = [utt_diag[k].get('raw_can_cosine') for k in valid_keys
                          if k in utt_diag]
    raw_can_l2_values = [utt_diag[k].get('raw_can_l2') for k in valid_keys
                         if k in utt_diag]
    if not raw_can_cos_values:
        raw_can_cos_values = [
            1.0 - _cosine_distance(raw_embs[k], can_embs[k])
            for k in valid_keys
        ]
    if not raw_can_l2_values:
        raw_can_l2_values = [
            float(np.linalg.norm(raw_embs[k] - can_embs[k]))
            for k in valid_keys
        ]

    gate_mean, gate_std = _mean_std(gate_values)
    uncertainty_mean, uncertainty_std = _mean_std(uncertainty_values)
    entropy_mean, entropy_std = _mean_std(entropy_values)
    residual_mean, residual_std = _mean_std(residual_values)
    raw_can_cos_mean, _ = _mean_std(raw_can_cos_values)
    raw_can_l2_mean, _ = _mean_std(raw_can_l2_values)

    report = {
        'mode': args.mode,
        'config': args.config,
        'checkpoint': args.checkpoint,
        'oracle_age_used': False,
        'raw_embedding_scp': args.raw_embedding_scp,
        'canonical_embedding_scp': args.canonical_embedding_scp,
        'num_common_embeddings': len(keys),
        'num_valid_age_embeddings': len(valid_keys),
        'same_speaker_cross_age_distance_raw': same_cross['raw_mean'],
        'same_speaker_cross_age_distance_canonical':
        same_cross['canonical_mean'],
        'same_speaker_cross_age_delta': same_cross['delta'],
        'same_speaker_cross_age_pair_count': same_cross['count'],
        'different_speaker_distance_raw': different['raw_mean'],
        'different_speaker_distance_canonical': different['canonical_mean'],
        'different_speaker_distance_delta': different['delta'],
        'different_speaker_pair_count': different['count'],
        'same_speaker_same_age_distance_raw': same_same['raw_mean'],
        'same_speaker_same_age_distance_canonical':
        same_same['canonical_mean'],
        'same_speaker_same_age_delta': same_same['delta'],
        'same_speaker_same_age_pair_count': same_same['count'],
        'path_valid_pair_count': same_cross['count'],
        'path_nonzero_batch_ratio': _read_path_ratio(args.train_log),
        'gate_mean': gate_mean if gate_mean is not None else
        diag_json.get('gate_mean'),
        'gate_std': gate_std if gate_std is not None else diag_json.get(
            'gate_std'),
        'gate_min': float(np.min(gate_values)) if gate_values else
        diag_json.get('gate_min'),
        'gate_max': float(np.max(gate_values)) if gate_values else
        diag_json.get('gate_max'),
        'uncertainty_mean': uncertainty_mean
        if uncertainty_mean is not None else diag_json.get('uncertainty_mean'),
        'uncertainty_std': uncertainty_std,
        'age_posterior_entropy_mean': entropy_mean
        if entropy_mean is not None else
        diag_json.get('age_posterior_entropy_mean'),
        'age_posterior_entropy_std': entropy_std,
        'residual_norm_mean': residual_mean if residual_mean is not None else
        diag_json.get('residual_norm_mean'),
        'residual_norm_std': residual_std,
        'raw_can_cosine_mean': raw_can_cos_mean
        if raw_can_cos_mean is not None else diag_json.get(
            'raw_can_cosine_mean'),
        'raw_can_l2_mean': raw_can_l2_mean
        if raw_can_l2_mean is not None else diag_json.get('raw_can_l2_mean'),
        'age_gap_bucket_metrics':
        _age_gap_metrics(buckets['same_cross'], raw_embs, can_embs, ages),
        'sampled_pair_counts': {
            'same_speaker_cross_age': len(buckets['same_cross']),
            'same_speaker_same_age': len(buckets['same_same_age']),
            'different_speaker': len(buckets['different']),
        },
        'interpretation_rules': [
            'same-speaker cross-age distance down while different-speaker '
            'distance is not clearly down supports the trajectory claim.',
            'both same-speaker and different-speaker distances down suggests '
            'possible embedding collapse or global contraction.',
            'same-speaker cross-age distance nearly unchanged suggests a near '
            'identity canonicalizer.',
            'lambda_path>0 must improve over lambda_path=0 before path '
            'consistency can be emphasized as learning an effective trajectory.',
            'This diagnostic supports claim analysis but is not an EER result.',
        ],
        'notes': [
            'Delta is canonical_mean - raw_mean; negative means distance was '
            'reduced by canonicalization.',
            'Age labels are used only for grouping/statistics, never passed to '
            'model forward.',
        ],
    }
    unavailable = {}
    non_metric_keys = {
        'mode', 'config', 'checkpoint', 'oracle_age_used',
        'raw_embedding_scp', 'canonical_embedding_scp', 'notes',
        'interpretation_rules', 'sampled_pair_counts', 'age_gap_bucket_metrics'
    }
    for key, value in report.items():
        if key in non_metric_keys:
            continue
        if value is None:
            unavailable[key] = 'metric could not be computed from provided inputs'
    if report['path_nonzero_batch_ratio'] is None:
        unavailable['path_nonzero_batch_ratio'] = (
            'no --train-log was provided or no path statistics were found')
    if args.mode == 'embedding' and not utt_diag and not args.diagnostic_json:
        for name in [
                'gate_mean', 'gate_std', 'gate_min', 'gate_max',
                'uncertainty_mean', 'uncertainty_std',
                'age_posterior_entropy_mean', 'age_posterior_entropy_std',
                'residual_norm_mean', 'residual_norm_std',
                'raw_can_l2_mean'
        ]:
            unavailable[name] = (
                'embedding mode has no internal ACSM state unless '
                '--diagnostic-json is provided')
    warnings = []
    for name, count in report['sampled_pair_counts'].items():
        if count == 0:
            warnings.append('no pairs available for {}'.format(name))
    warnings.extend([
        '{} unavailable: {}'.format(k, v)
        for k, v in sorted(unavailable.items())
    ])
    report['unavailable_metrics'] = unavailable
    report['warnings'] = warnings
    return report, valid_keys, utt2spk, ages


def main():
    parser = argparse.ArgumentParser(
        description='Diagnose ACSM canonical trajectory behavior.')
    parser.add_argument('--mode', choices=['embedding', 'model'],
                        default='embedding',
                        help='embedding analyzes existing embeddings; model '
                        'loads ACSM and extracts raw/canonical embeddings.')
    parser.add_argument('--config',
                        help='Required in model mode; metadata in embedding mode.')
    parser.add_argument('--checkpoint',
                        help='Required in model mode; metadata in embedding mode.')
    parser.add_argument('--utt2spk', required=True)
    parser.add_argument('--age-label-file', required=True)
    parser.add_argument('--age-label-type', choices=['value', 'group'],
                        default='value')
    parser.add_argument('--age-bins', default='21,31,41,51,61,71')
    parser.add_argument('--ignore-age-index', type=int, default=-1)
    parser.add_argument('--raw-embedding-scp',
                        help='Embedding mode raw observed embedding scp.')
    parser.add_argument('--canonical-embedding-scp',
                        help='Embedding mode canonical embedding scp.')
    parser.add_argument('--wav-scp',
                        help='Model mode kaldi wav.scp: utt path-or-command.')
    parser.add_argument('--data-list',
                        help='Model mode WeSpeaker raw/feat JSON list.')
    parser.add_argument('--data-type', choices=['raw', 'feat'], default='raw')
    parser.add_argument('--utt-list',
                        help='Optional one-column utterance filter for wav.scp.')
    parser.add_argument('--trial-file')
    parser.add_argument('--diagnostic-json',
                        help='Optional diagnose_acsm.py JSON for embedding mode.')
    parser.add_argument('--train-log')
    parser.add_argument('--max-utts', type=int)
    parser.add_argument('--max-pairs', type=int, default=200000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-utterance-diagnostics')
    parser.add_argument('--save-embeddings-dir')
    parser.add_argument('--output-json')
    args = parser.parse_args()

    if args.mode == 'embedding':
        if not args.raw_embedding_scp or not args.canonical_embedding_scp:
            raise ValueError('embedding mode requires --raw-embedding-scp and '
                             '--canonical-embedding-scp')
        raw_embs = _read_embeddings(args.raw_embedding_scp, args.max_utts)
        can_embs = _read_embeddings(args.canonical_embedding_scp,
                                    args.max_utts)
        utt_diag = {}
    else:
        if not args.config or not args.checkpoint:
            raise ValueError('model mode requires --config and --checkpoint')
        raw_embs, can_embs, utt_diag = _extract_model_mode(args)

    report, valid_keys, utt2spk, ages = _build_report(args, raw_embs, can_embs,
                                                      utt_diag)
    if args.save_utterance_diagnostics:
        _save_utterance_diagnostics(args.save_utterance_diagnostics,
                                    valid_keys, utt2spk, ages, utt_diag)
    if args.save_embeddings_dir:
        _save_embeddings(args.save_embeddings_dir, valid_keys, raw_embs,
                         can_embs, utt_diag)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)),
                    exist_ok=True)
        with open(args.output_json, 'w') as f:
            f.write(text + '\n')


if __name__ == '__main__':
    main()
