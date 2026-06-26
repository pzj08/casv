#!/usr/bin/env python3

import argparse
import json
import random

import numpy as np


def _read_two_col(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                rows.append((parts[0], parts[1]))
    return rows


def _age_value_to_group(value, bins):
    value = float(value)
    for idx, boundary in enumerate(bins):
        if value < float(boundary):
            return idx
    return len(bins)


def _load_age_labels(path, age_label_type, bins, ignore_index):
    if path.endswith('.npy'):
        loaded = np.load(path, allow_pickle=True)
        if getattr(loaded, 'shape', None) == ():
            loaded = loaded.item()
        rows = loaded.items()
    else:
        rows = _read_two_col(path)
    labels = {}
    for key, value in rows:
        labels[key] = (_age_value_to_group(value, bins)
                       if age_label_type == 'value' else int(value))
    return labels


def _lookup_age(key, labels, ignore_index):
    candidates = [key]
    if key.endswith('.wav'):
        candidates.append(key[:-4])
    if '/' in key:
        parts = key[:-4].split('/') if key.endswith('.wav') else key.split('/')
        if len(parts) >= 2:
            candidates.append('-'.join(parts[:2]))
    if '-' in key:
        candidates.append(key.rsplit('-', 1)[0])
    for candidate in candidates:
        if candidate in labels:
            return labels[candidate]
    return ignore_index


def _pair_count(batch, ignore_index):
    count = 0
    valid_count = sum(1 for _, _, age in batch if age != ignore_index)
    for i in range(len(batch)):
        for j in range(i + 1, len(batch)):
            if (batch[i][1] == batch[j][1] and batch[i][2] != ignore_index
                    and batch[j][2] != ignore_index
                    and batch[i][2] != batch[j][2]):
                count += 1
    return valid_count, count


def main():
    parser = argparse.ArgumentParser(
        description='Estimate same-speaker different-age pair coverage.')
    parser.add_argument('--utt2spk', required=True)
    parser.add_argument('--age-label-file', required=True)
    parser.add_argument('--age-label-type', default='value')
    parser.add_argument('--age-bins', default='21,31,41,51,61,71')
    parser.add_argument('--batch-size', type=int, default=96)
    parser.add_argument('--num-batches', type=int, default=200)
    parser.add_argument('--ignore-age-index', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--output-json')
    args = parser.parse_args()

    bins = [float(x) for x in args.age_bins.split(',') if x]
    labels = _load_age_labels(args.age_label_file, args.age_label_type, bins,
                              args.ignore_age_index)
    rows = [(utt, spk, _lookup_age(utt, labels, args.ignore_age_index))
            for utt, spk in _read_two_col(args.utt2spk)]
    rng = random.Random(args.seed)
    pair_counts = []
    valid_counts = []
    for _ in range(args.num_batches):
        batch = rng.sample(rows, min(args.batch_size, len(rows)))
        valid, pairs = _pair_count(batch, args.ignore_age_index)
        valid_counts.append(valid)
        pair_counts.append(pairs)
    nonzero = sum(1 for x in pair_counts if x > 0)
    nonzero_ratio = nonzero / float(max(len(pair_counts), 1))
    report = {
        'batch_size': args.batch_size,
        'num_batches': args.num_batches,
        'valid_age_sample_mean': sum(valid_counts) / float(len(valid_counts)),
        'path_valid_pair_count_mean':
        sum(pair_counts) / float(len(pair_counts)),
        'path_valid_pair_count_max': max(pair_counts) if pair_counts else 0,
        'path_nonzero_batch_ratio': nonzero_ratio,
        'recommendation':
        ('Path coverage is low; consider an age-aware sampler before relying on lambda_path.'
         if nonzero_ratio < 0.2 else
         'Random batches have enough path pairs for weak lambda_path ablation.'),
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        with open(args.output_json, 'w') as f:
            f.write(text + '\n')


if __name__ == '__main__':
    main()
