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


def _load_age_label_map(path):
    if path.endswith('.npy') or path.endswith('.npz'):
        loaded = np.load(path, allow_pickle=True)
        if isinstance(loaded, np.ndarray) and loaded.shape == ():
            loaded = loaded.item()
        if isinstance(loaded, np.lib.npyio.NpzFile):
            loaded = {k: loaded[k] for k in loaded.files}
        if not isinstance(loaded, dict):
            raise ValueError('age_label_file must be a dict-like npy/npz file')
        return loaded
    loaded = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                loaded[parts[0]] = parts[1]
    return loaded


def _coerce_age_value(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        value = value[0]
    return float(value)


def _read_age_labels_with_values(path, keys, label_type, bins, ignore_index):
    loaded = _load_age_label_map(path)
    age_groups = {}
    age_values = {}
    for key in keys:
        value = None
        for cand in _key_candidates(key):
            if cand in loaded:
                value = loaded[cand]
                break
        if value is None:
            age_groups[key] = ignore_index
            age_values[key] = None
            continue
        value = _coerce_age_value(value)
        if label_type == 'value':
            age_groups[key] = _age_value_to_group(value, bins)
            age_values[key] = value
        else:
            age_groups[key] = int(value)
            age_values[key] = None
    return age_groups, age_values


def _read_age_labels(path, keys, label_type, bins, ignore_index):
    age_groups, _ = _read_age_labels_with_values(path, keys, label_type, bins,
                                                 ignore_index)
    return age_groups


def _read_embeddings(path, max_utts=None):
    out = {}
    for key, emb in kaldiio.load_scp_sequential(path):
        out[key] = np.asarray(emb, dtype=np.float32)
        if max_utts and len(out) >= max_utts:
            break
    return out


def _read_utt_list(path, max_utts=None):
    utts = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            utts.append(parts[0])
            if max_utts and len(utts) >= max_utts:
                break
    return utts


def _read_embedding_array(path, utt_list, max_utts=None):
    if not utt_list:
        raise ValueError('npy embedding input requires --utt-list')
    keys = _read_utt_list(utt_list, max_utts)
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 2:
        raise ValueError('embedding npy must be a 2-D array')
    if len(keys) > arr.shape[0]:
        raise ValueError('--utt-list has more entries than embedding rows')
    return {
        key: np.asarray(arr[idx], dtype=np.float32)
        for idx, key in enumerate(keys)
    }


def _read_embedding_input(path, utt_list=None, max_utts=None):
    if path.endswith('.npy'):
        return _read_embedding_array(path, utt_list, max_utts)
    return _read_embeddings(path, max_utts)


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


def _safe_mean(values):
    if len(values) == 0:
        return None
    return float(np.mean(values))


def _safe_std(values):
    if len(values) == 0:
        return None
    return float(np.std(values))


def _safe_percentile(values, pct):
    if len(values) == 0:
        return None
    return float(np.percentile(values, pct))


def _safe_relative(delta, raw_mean, eps=1.0e-12):
    if delta is None or raw_mean is None or abs(raw_mean) <= eps:
        return None
    return float(delta / raw_mean)


def _bool_or_none(value):
    if value is None:
        return None
    return bool(value)


def _cosine_similarity_np(a, b, eps=1.0e-12):
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), eps)
    return float(np.dot(a, b) / denom)


def _normalize_np(x, eps=1.0e-12):
    norm = max(float(np.linalg.norm(x)), eps)
    return x / norm


def _embedding_change_metrics(keys, raw_embs, can_embs, identity_cos_threshold,
                              identity_delta_threshold):
    cos_values = []
    delta_raw_l2 = []
    delta_norm_l2 = []
    for key in keys:
        raw = np.asarray(raw_embs[key], dtype=np.float32)
        can = np.asarray(can_embs[key], dtype=np.float32)
        cos_values.append(_cosine_similarity_np(raw, can))
        delta_raw_l2.append(float(np.linalg.norm(can - raw)))
        delta_norm_l2.append(float(np.linalg.norm(_normalize_np(can) -
                                                  _normalize_np(raw))))
    cos_mean = _safe_mean(cos_values)
    delta_norm_mean = _safe_mean(delta_norm_l2)
    identity_like = None
    if cos_mean is not None and delta_norm_mean is not None:
        identity_like = (cos_mean >= identity_cos_threshold
                         and delta_norm_mean <= identity_delta_threshold)
    return {
        'raw_can_cosine_mean': cos_mean,
        'raw_can_cosine_std': _safe_std(cos_values),
        'raw_can_cosine_min': float(np.min(cos_values)) if cos_values else None,
        'raw_can_cosine_p05': _safe_percentile(cos_values, 5),
        'raw_can_cosine_p50': _safe_percentile(cos_values, 50),
        'raw_can_cosine_p95': _safe_percentile(cos_values, 95),
        'delta_raw_l2_mean': _safe_mean(delta_raw_l2),
        'delta_raw_l2_std': _safe_std(delta_raw_l2),
        'delta_norm_l2_mean': delta_norm_mean,
        'delta_norm_l2_std': _safe_std(delta_norm_l2),
        'identity_like': _bool_or_none(identity_like),
    }


def _pair_distance_arrays(pairs, raw_embs, can_embs):
    raw = []
    can = []
    for a, b in pairs:
        raw.append(_cosine_distance(raw_embs[a], raw_embs[b]))
        can.append(_cosine_distance(can_embs[a], can_embs[b]))
    raw = np.asarray(raw, dtype=np.float64)
    can = np.asarray(can, dtype=np.float64)
    return raw, can, raw - can


def _bootstrap_ci(values, bootstrap, seed):
    values = np.asarray(values, dtype=np.float64)
    if bootstrap <= 0 or values.size == 0:
        return None
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(bootstrap):
        sample = values[rng.integers(0, values.size, size=values.size)]
        means.append(float(np.mean(sample)))
    return [float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5))]


def _bootstrap_ratio_ci(indicators, bootstrap, seed):
    indicators = np.asarray(indicators, dtype=np.float64)
    if bootstrap <= 0 or indicators.size == 0:
        return None
    return _bootstrap_ci(indicators, bootstrap, seed)


def _effectiveness_pair_metrics(pairs,
                                raw_embs,
                                can_embs,
                                bootstrap=0,
                                seed=1234,
                                bad_compress_threshold=0.01,
                                include_bad_compress=False):
    raw, can, delta = _pair_distance_arrays(pairs, raw_embs, can_embs)
    if raw.size == 0:
        out = {
            'raw_distance_mean': None,
            'canonical_distance_mean': None,
            'delta_mean_raw_minus_can': None,
            'delta_relative': None,
            'num_pairs': 0,
        }
        if include_bad_compress:
            out['bad_compress_ratio'] = None
        else:
            out['improved_pair_ratio'] = None
        return out
    raw_mean = float(np.mean(raw))
    can_mean = float(np.mean(can))
    delta_mean = float(np.mean(delta))
    out = {
        'raw_distance_mean': raw_mean,
        'canonical_distance_mean': can_mean,
        'delta_mean_raw_minus_can': delta_mean,
        'delta_relative': _safe_relative(delta_mean, raw_mean),
        'num_pairs': int(raw.size),
    }
    if include_bad_compress:
        out['bad_compress_ratio'] = float(
            np.mean(delta > bad_compress_threshold))
    else:
        improved = delta > 0.0
        out['improved_pair_ratio'] = float(np.mean(improved))
    if bootstrap > 0:
        out['delta_mean_raw_minus_can_ci95'] = _bootstrap_ci(
            delta, bootstrap, seed)
        if not include_bad_compress:
            out['improved_pair_ratio_ci95'] = _bootstrap_ratio_ci(
                delta > 0.0, bootstrap, seed + 17)
    return out


def _same_age_pair_metrics(pairs, raw_embs, can_embs):
    raw, can, delta = _pair_distance_arrays(pairs, raw_embs, can_embs)
    if raw.size == 0:
        return {
            'raw_distance_mean': None,
            'canonical_distance_mean': None,
            'delta_mean_raw_minus_can': None,
            'num_pairs': 0,
        }
    return {
        'raw_distance_mean': float(np.mean(raw)),
        'canonical_distance_mean': float(np.mean(can)),
        'delta_mean_raw_minus_can': float(np.mean(delta)),
        'num_pairs': int(raw.size),
    }


def _parse_age_gap_buckets(value):
    return [float(x) for x in value.split(',') if x.strip()]


def _format_gap_key(gap):
    if float(gap).is_integer():
        return 'gap_ge_{}'.format(int(gap))
    return 'gap_ge_{}'.format(str(gap).replace('.', 'p'))


def _age_gap_bucket_effectiveness(pairs, raw_embs, can_embs, age_values,
                                  age_gap_buckets):
    out = {}
    for gap in age_gap_buckets:
        key = _format_gap_key(gap)
        gap_pairs = []
        for a, b in pairs:
            age_a = age_values.get(a)
            age_b = age_values.get(b)
            if age_a is None or age_b is None:
                continue
            if abs(float(age_a) - float(age_b)) >= gap:
                gap_pairs.append((a, b))
        metrics = _effectiveness_pair_metrics(gap_pairs, raw_embs, can_embs)
        out[key] = {
            'num_pairs': metrics['num_pairs'],
            'raw_distance_mean': metrics['raw_distance_mean'],
            'canonical_distance_mean': metrics['canonical_distance_mean'],
            'delta_mean_raw_minus_can':
            metrics['delta_mean_raw_minus_can'],
            'improved_pair_ratio': metrics.get('improved_pair_ratio'),
            'unreliable': metrics['num_pairs'] == 0,
        }
    return out


def _add_effectiveness_pair(buckets, a, b, utt2spk, age_groups, ignore_index,
                            max_same_pairs):
    if a not in utt2spk or b not in utt2spk:
        return
    age_a = age_groups.get(a, ignore_index)
    age_b = age_groups.get(b, ignore_index)
    same = utt2spk[a] == utt2spk[b]
    if not same:
        return
    if age_a == ignore_index or age_b == ignore_index:
        return
    name = 'same_same_age' if age_a == age_b else 'same_cross'
    if len(buckets[name]) < max_same_pairs:
        buckets[name].append((a, b))


def _sample_effectiveness_pairs(keys, utt2spk, age_groups, ignore_index,
                                max_same_pairs, max_diff_pairs, seed,
                                trial_file=None):
    rng = random.Random(seed)
    keys = list(keys)
    key_set = set(keys)
    buckets = {'same_cross': [], 'same_same_age': [], 'different': []}
    if trial_file:
        for a, b in _parse_trial_pairs(trial_file):
            if a not in key_set or b not in key_set:
                continue
            if utt2spk.get(a) == utt2spk.get(b):
                _add_effectiveness_pair(buckets, a, b, utt2spk, age_groups,
                                        ignore_index, max_same_pairs)
            elif len(buckets['different']) < max_diff_pairs:
                buckets['different'].append((a, b))
        return buckets

    by_spk = defaultdict(list)
    for key in keys:
        if key in utt2spk:
            by_spk[utt2spk[key]].append(key)
    spk_ids = list(by_spk)
    rng.shuffle(spk_ids)
    for spk in spk_ids:
        utts = list(by_spk[spk])
        rng.shuffle(utts)
        stop_spk = False
        for i in range(len(utts)):
            for j in range(i + 1, len(utts)):
                _add_effectiveness_pair(buckets, utts[i], utts[j], utt2spk,
                                        age_groups, ignore_index,
                                        max_same_pairs)
                if (len(buckets['same_cross']) >= max_same_pairs
                        and len(buckets['same_same_age']) >= max_same_pairs):
                    stop_spk = True
                    break
            if stop_spk:
                break
        if (len(buckets['same_cross']) >= max_same_pairs
                and len(buckets['same_same_age']) >= max_same_pairs):
            break

    if len(keys) >= 2 and max_diff_pairs > 0:
        seen = set()
        attempts = 0
        max_attempts = max(max_diff_pairs * 100, 1000)
        while len(buckets['different']) < max_diff_pairs and attempts < max_attempts:
            a, b = rng.sample(keys, 2)
            attempts += 1
            if utt2spk.get(a) == utt2spk.get(b):
                continue
            pair = tuple(sorted((a, b)))
            if pair in seen:
                continue
            seen.add(pair)
            buckets['different'].append((a, b))
    return buckets


def _effectiveness_decision(embedding_change, same_cross_age,
                            different_speaker, identity_cos_threshold,
                            identity_delta_threshold):
    cos_mean = embedding_change.get('raw_can_cosine_mean')
    delta_norm = embedding_change.get('delta_norm_l2_mean')
    embedding_changed = None
    near_identity_risk = None
    if cos_mean is not None and delta_norm is not None:
        embedding_changed = (cos_mean < identity_cos_threshold
                             or delta_norm > identity_delta_threshold)
        near_identity_risk = (cos_mean >= identity_cos_threshold
                              and delta_norm <= identity_delta_threshold)

    same_delta = same_cross_age.get('delta_mean_raw_minus_can')
    same_ratio = same_cross_age.get('improved_pair_ratio')
    if same_cross_age.get('num_pairs', 0) == 0:
        same_improved = None
    else:
        same_improved = same_delta is not None and same_delta > 0.0 and (
            same_ratio is not None and same_ratio > 0.5)

    diff_rel = different_speaker.get('delta_relative')
    if different_speaker.get('num_pairs', 0) == 0 or diff_rel is None:
        diff_preserved = None
        collapse_risk = None
    else:
        diff_preserved = diff_rel < 0.02
        collapse_risk = diff_rel >= 0.05

    if (near_identity_risk is True or same_improved is False
            or collapse_risk is True or same_improved is None):
        overall = 'FAIL'
    elif (embedding_changed is True and same_improved is True
          and diff_preserved is True and collapse_risk is False):
        overall = 'PASS'
    elif same_improved is True:
        overall = 'PARTIAL'
    else:
        overall = 'FAIL'

    return {
        'embedding_changed': _bool_or_none(embedding_changed),
        'same_cross_age_improved': _bool_or_none(same_improved),
        'different_speaker_preserved': _bool_or_none(diff_preserved),
        'collapse_risk': _bool_or_none(collapse_risk),
        'near_identity_risk': _bool_or_none(near_identity_risk),
        'overall': overall,
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
    ages, age_values = _read_age_labels_with_values(args.age_label_file, keys,
                                                    args.age_label_type, bins,
                                                    args.ignore_age_index)
    valid_keys = [
        k for k in keys
        if k in utt2spk and ages.get(k, args.ignore_age_index) !=
        args.ignore_age_index
    ]
    if args.max_utts:
        valid_keys = valid_keys[:args.max_utts]
    max_same_pairs = getattr(args, 'max_same_pairs',
                             getattr(args, 'max_pairs', 200000))
    max_diff_pairs = getattr(args, 'max_diff_pairs',
                             getattr(args, 'max_pairs', 200000))
    seed = getattr(args, 'seed', 1234)
    buckets = _sample_effectiveness_pairs(valid_keys, utt2spk, ages,
                                          args.ignore_age_index,
                                          max_same_pairs, max_diff_pairs, seed,
                                          args.trial_file)
    same_cross = _pair_stats(buckets['same_cross'], raw_embs, can_embs)
    same_same = _pair_stats(buckets['same_same_age'], raw_embs, can_embs)
    different = _pair_stats(buckets['different'], raw_embs, can_embs)

    identity_cos_threshold = getattr(args, 'identity_cos_threshold', 0.9999)
    identity_delta_threshold = getattr(args, 'identity_delta_threshold',
                                       1.0e-5)
    collapse_diff_threshold = getattr(args, 'collapse_diff_threshold', 0.01)
    bootstrap = getattr(args, 'bootstrap', 0) or 0
    age_gap_buckets = _parse_age_gap_buckets(
        getattr(args, 'age_gap_buckets', '5,10,15,20'))

    embedding_change = _embedding_change_metrics(
        valid_keys, raw_embs, can_embs, identity_cos_threshold,
        identity_delta_threshold)
    same_cross_effect = _effectiveness_pair_metrics(
        buckets['same_cross'],
        raw_embs,
        can_embs,
        bootstrap=bootstrap,
        seed=seed,
        bad_compress_threshold=collapse_diff_threshold)
    different_effect = _effectiveness_pair_metrics(
        buckets['different'],
        raw_embs,
        can_embs,
        bootstrap=bootstrap,
        seed=seed + 101,
        bad_compress_threshold=collapse_diff_threshold,
        include_bad_compress=True)
    same_same_effect = _same_age_pair_metrics(buckets['same_same_age'],
                                              raw_embs, can_embs)
    age_gap_effect = _age_gap_bucket_effectiveness(buckets['same_cross'],
                                                   raw_embs, can_embs,
                                                   age_values,
                                                   age_gap_buckets)
    decision = _effectiveness_decision(embedding_change, same_cross_effect,
                                       different_effect,
                                       identity_cos_threshold,
                                       identity_delta_threshold)

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
        'raw_embeddings': getattr(args, 'raw_embeddings', None),
        'canonical_embeddings': getattr(args, 'canonical_embeddings', None),
        'num_common_embeddings': len(keys),
        'num_valid_age_embeddings': len(valid_keys),
        'num_same_same_age_pairs': len(buckets['same_same_age']),
        'num_same_cross_age_pairs': len(buckets['same_cross']),
        'num_diff_speaker_pairs': len(buckets['different']),
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
        'embedding_change': embedding_change,
        'same_speaker_cross_age': same_cross_effect,
        'different_speaker': different_effect,
        'same_speaker_same_age': same_same_effect,
        'age_gap_buckets': age_gap_effect,
        'effectiveness_decision': decision,
        'effectiveness_thresholds': {
            'identity_cos_threshold': identity_cos_threshold,
            'identity_delta_threshold': identity_delta_threshold,
            'collapse_diff_threshold': collapse_diff_threshold,
            'different_speaker_preserved_delta_relative_threshold': 0.02,
            'collapse_delta_relative_threshold': 0.05,
        },
        'sampled_pair_counts': {
            'same_speaker_cross_age': len(buckets['same_cross']),
            'same_speaker_same_age': len(buckets['same_same_age']),
            'different_speaker': len(buckets['different']),
            'num_same_cross_age_pairs': len(buckets['same_cross']),
            'num_same_same_age_pairs': len(buckets['same_same_age']),
            'num_diff_speaker_pairs': len(buckets['different']),
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
            'Effectiveness report uses raw - canonical; positive means '
            'canonicalization reduced the distance.',
            'Age labels are used only for grouping/statistics, never passed to '
            'model forward.',
        ],
    }
    for gap_key, values in age_gap_effect.items():
        count_key = 'num_pairs_{}'.format(gap_key)
        report[count_key] = values['num_pairs']
        report['sampled_pair_counts'][count_key] = values['num_pairs']
    unavailable = {}
    non_metric_keys = {
        'mode', 'config', 'checkpoint', 'oracle_age_used',
        'raw_embedding_scp', 'canonical_embedding_scp', 'notes',
        'raw_embeddings', 'canonical_embeddings', 'interpretation_rules',
        'sampled_pair_counts', 'age_gap_bucket_metrics', 'embedding_change',
        'same_speaker_cross_age', 'different_speaker',
        'same_speaker_same_age', 'age_gap_buckets',
        'effectiveness_decision', 'effectiveness_thresholds'
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
        if name.startswith('num_pairs_'):
            continue
        if count == 0:
            warnings.append('no pairs available for {}'.format(name))
    for gap_key, values in age_gap_effect.items():
        if values['num_pairs'] == 0:
            warnings.append('no same-speaker pairs available for {}'.format(
                gap_key))
    if args.age_label_type != 'value':
        warnings.append('age gap buckets require continuous age labels; '
                        'bucket metrics are unreliable with group labels')
    if decision['near_identity_risk'] is True:
        warnings.append('ACSM appears near-identity under current thresholds')
    if decision['collapse_risk'] is True:
        warnings.append('different-speaker distances are compressed enough to '
                        'flag collapse risk')
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
    parser.add_argument('--raw-embeddings',
                        help='Embedding mode raw embeddings .npy array.')
    parser.add_argument('--canonical-embeddings',
                        help='Embedding mode canonical embeddings .npy array.')
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
    parser.add_argument('--effectiveness-report', action='store_true',
                        help='Emit ACSM pair-level effectiveness diagnostics.')
    parser.add_argument('--age-gap-buckets', default='5,10,15,20')
    parser.add_argument('--max-utts', type=int)
    parser.add_argument('--max-pairs', type=int, default=200000)
    parser.add_argument('--max-same-pairs', type=int, default=200000)
    parser.add_argument('--max-diff-pairs', type=int, default=200000)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--collapse-diff-threshold',
                        type=float,
                        default=0.01)
    parser.add_argument('--identity-cos-threshold',
                        type=float,
                        default=0.9999)
    parser.add_argument('--identity-delta-threshold',
                        type=float,
                        default=1.0e-5)
    parser.add_argument('--bootstrap', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-utterance-diagnostics')
    parser.add_argument('--save-embeddings-dir')
    parser.add_argument('--output-json')
    args = parser.parse_args()

    if args.mode == 'embedding':
        raw_path = args.raw_embeddings or args.raw_embedding_scp
        can_path = args.canonical_embeddings or args.canonical_embedding_scp
        if not raw_path or not can_path:
            raise ValueError('embedding mode requires --raw-embeddings/'
                             '--raw-embedding-scp and --canonical-embeddings/'
                             '--canonical-embedding-scp')
        raw_embs = _read_embedding_input(raw_path, args.utt_list,
                                         args.max_utts)
        can_embs = _read_embedding_input(can_path, args.utt_list,
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
