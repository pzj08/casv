#!/usr/bin/env python3

import argparse
import json
import os

import numpy as np


def _read_two_col(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                rows.append((parts[0], parts[1]))
    return rows


def _read_utts_from_wav_or_list(path):
    utts = set()
    with open(path) as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            if text.startswith('{'):
                import json as _json
                utts.add(_json.loads(text)['key'])
            else:
                utts.add(text.split()[0])
    return utts


def _speaker_from_utt(utt):
    if '/' in utt:
        return utt.split('/')[0]
    if '-' in utt:
        return utt.split('-')[0]
    return utt


def _age_label_keys(path):
    if not path:
        return set()
    if path.endswith('.npy'):
        loaded = np.load(path, allow_pickle=True)
        if getattr(loaded, 'shape', None) == ():
            loaded = loaded.item()
        return set(loaded.keys())
    return {row[0] for row in _read_two_col(path)}


def _trial_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                rows.append((parts[0], parts[1], parts[2]))
    return rows


def _short(items, n=20):
    return sorted(list(items))[:n]


def main():
    parser = argparse.ArgumentParser(
        description='Audit speaker/utterance/trial/age-label leakage.')
    parser.add_argument('--train-utt2spk', required=True)
    parser.add_argument('--train-list')
    parser.add_argument('--eval-utt-list', required=True)
    parser.add_argument('--eval-trial', action='append', required=True)
    parser.add_argument('--age-label-file')
    parser.add_argument('--allow-speaker-overlap', action='store_true')
    parser.add_argument('--output-json')
    args = parser.parse_args()

    train_pairs = _read_two_col(args.train_utt2spk)
    train_utts = {u for u, _ in train_pairs}
    train_speakers = {s for _, s in train_pairs}
    if args.train_list:
        train_utts |= _read_utts_from_wav_or_list(args.train_list)

    eval_utts = _read_utts_from_wav_or_list(args.eval_utt_list)
    eval_speakers = {_speaker_from_utt(u) for u in eval_utts}
    labels = _age_label_keys(args.age_label_file)

    all_trial_utts = set()
    trial_reports = {}
    legal_labels = {'target', 'nontarget', '1', '0', 'true', 'false'}
    for trial in args.eval_trial:
        rows = _trial_rows(trial)
        e1 = {r[0] for r in rows}
        e2 = {r[1] for r in rows}
        all_trial_utts |= e1 | e2
        labels_seen = [r[2].lower() for r in rows]
        invalid = [r for r in rows if r[2].lower() not in legal_labels]
        self_trials = [r for r in rows if r[0] == r[1]]
        positives = sum(1 for x in labels_seen if x in ('target', '1', 'true'))
        negatives = len(rows) - positives
        missing = (e1 | e2) - eval_utts
        trial_reports[trial] = {
            'num_trials': len(rows),
            'positive_count': positives,
            'negative_count': negatives,
            'invalid_label_count': len(invalid),
            'self_trial_count': len(self_trials),
            'missing_eval_utt_count': len(missing),
            'missing_eval_utt_examples': _short(missing),
        }

    speaker_overlap = train_speakers & eval_speakers
    utt_overlap = train_utts & eval_utts
    age_eval_overlap = labels & all_trial_utts
    age_train_overlap = labels & train_utts

    fail = bool(utt_overlap)
    high_risk = bool(speaker_overlap) and not args.allow_speaker_overlap
    trial_fail = any(v['invalid_label_count'] or v['self_trial_count']
                     or v['missing_eval_utt_count']
                     for v in trial_reports.values())
    report = {
        'status':
        'FAIL' if (fail or high_risk or trial_fail) else 'PASS',
        'train_speaker_count':
        len(train_speakers),
        'eval_speaker_count':
        len(eval_speakers),
        'train_eval_speaker_overlap_count':
        len(speaker_overlap),
        'train_eval_speaker_overlap_examples':
        _short(speaker_overlap),
        'train_eval_utt_overlap_count':
        len(utt_overlap),
        'train_eval_utt_overlap_examples':
        _short(utt_overlap),
        'age_label_eval_trial_utt_overlap_count':
        len(age_eval_overlap),
        'age_label_eval_trial_utt_overlap_examples':
        _short(age_eval_overlap),
        'age_label_train_utt_overlap_count':
        len(age_train_overlap),
        'age_label_train_utt_overlap_examples':
        _short(age_train_overlap),
        'trial_reports':
        trial_reports,
        'notes': [
            'Age labels covering eval utterances are not leakage unless the extraction/scoring path uses them.',
            'Speaker overlap is high risk by default; pass --allow-speaker-overlap only when the protocol explicitly permits it.',
        ],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        with open(args.output_json, 'w') as f:
            f.write(text + '\n')


if __name__ == '__main__':
    main()
