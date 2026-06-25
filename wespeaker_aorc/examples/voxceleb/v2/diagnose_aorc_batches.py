#!/usr/bin/env python

import argparse
import json
from collections import Counter

import torch
import yaml
from torch.utils.data import DataLoader

from wespeaker.bin.train import _load_age_labels
from wespeaker.dataset.dataset import Dataset
from wespeaker.models.aorc_modules import get_aorc_config
from wespeaker.utils.file_utils import read_table
from wespeaker.utils.utils import spk2id


def _pair_stats(speakers, age_group, ignore_index, large_gap=2):
    batch = speakers.numel()
    eye = torch.eye(batch, dtype=torch.bool)
    same_spk = (speakers.view(-1, 1) == speakers.view(1, -1)) & ~eye
    diff_spk = (speakers.view(-1, 1) != speakers.view(1, -1)) & ~eye
    valid_age = age_group != ignore_index
    both_valid = valid_age.view(-1, 1) & valid_age.view(1, -1)
    gap = (age_group.view(-1, 1) - age_group.view(1, -1)).abs()
    same_age = both_valid & (gap == 0)
    diff_age = both_valid & (gap > 0)
    cross_age = same_spk & diff_age
    large_gap_mask = same_spk & both_valid & (gap >= large_gap)
    same_gap = gap[same_spk & both_valid].float()
    return {
        'batch_size': batch,
        'valid_age_ratio': valid_age.float().mean().item(),
        'num_speakers_per_batch': speakers.unique().numel(),
        'same_speaker_pairs': same_spk.sum().item(),
        'same_speaker_cross_age_pairs': cross_age.sum().item(),
        'same_speaker_large_gap_pairs': large_gap_mask.sum().item(),
        'diff_speaker_same_age_pairs': (diff_spk & same_age).sum().item(),
        'diff_speaker_diff_age_pairs': (diff_spk & diff_age).sum().item(),
        'mean_age_gap_same_speaker':
        same_gap.mean().item() if same_gap.numel() > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='conf/baseline_resnet34_aorc_full.yaml')
    parser.add_argument('--data_type', required=True)
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--train_label', required=True)
    parser.add_argument('--num_batches', type=int, default=100)
    parser.add_argument('--large_gap', type=int, default=2)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        configs = yaml.safe_load(f)
    aorc_conf = get_aorc_config(configs)
    train_rows = read_table(args.train_label)
    spk2id_dict = spk2id(train_rows)
    age_labels = _load_age_labels(aorc_conf['age_label_file'],
                                  aorc_conf['age_label_type'],
                                  aorc_conf['age_bins'],
                                  aorc_conf['num_age_groups'],
                                  aorc_conf['ignore_age_index'])

    dataset = Dataset(args.data_type,
                      args.train_data,
                      configs['dataset_args'],
                      spk2id_dict,
                      key_filter_file=configs.get('key_filter_file', None),
                      age_labels=age_labels,
                      ignore_age_index=aorc_conf['ignore_age_index'])
    dataloader = DataLoader(dataset, **configs['dataloader_args'])

    totals = Counter()
    age_hist = Counter()
    gap_sum = 0.0
    batches = 0
    caa_positive_batches = 0
    dir_positive_batches = 0
    for batch in dataloader:
        speakers = batch.get('orig_label', batch['label']).long()
        age_group = batch.get(
            'age_group',
            torch.full_like(speakers, aorc_conf['ignore_age_index'])).long()
        stats = _pair_stats(speakers, age_group, aorc_conf['ignore_age_index'],
                            args.large_gap)
        for key, value in stats.items():
            if key != 'mean_age_gap_same_speaker':
                totals[key] += value
        gap_sum += stats['mean_age_gap_same_speaker']
        caa_positive_batches += int(stats['same_speaker_cross_age_pairs'] > 0)
        dir_positive_batches += int(stats['same_speaker_cross_age_pairs'] > 0)
        for value in age_group.tolist():
            age_hist[int(value)] += 1
        batches += 1
        if batches >= args.num_batches:
            break

    denom = max(batches, 1)
    summary = {
        'num_batches': batches,
        'batch_size': totals['batch_size'] / denom,
        'valid_age_ratio': totals['valid_age_ratio'] / denom,
        'num_speakers_per_batch': totals['num_speakers_per_batch'] / denom,
        'same_speaker_pairs': totals['same_speaker_pairs'] / denom,
        'same_speaker_cross_age_pairs':
        totals['same_speaker_cross_age_pairs'] / denom,
        'same_speaker_large_gap_pairs':
        totals['same_speaker_large_gap_pairs'] / denom,
        'diff_speaker_same_age_pairs':
        totals['diff_speaker_same_age_pairs'] / denom,
        'diff_speaker_diff_age_pairs':
        totals['diff_speaker_diff_age_pairs'] / denom,
        'mean_age_gap_same_speaker': gap_sum / denom,
        'fraction_batches_with_caa_positive': caa_positive_batches / denom,
        'fraction_batches_with_dir_positive': dir_positive_batches / denom,
        'age_group_histogram': dict(sorted(age_hist.items())),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
