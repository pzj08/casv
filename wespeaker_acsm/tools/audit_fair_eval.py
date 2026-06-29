#!/usr/bin/env python3

import argparse
import json
import os
import re

import yaml


def _check(name, passed, risk, detail, fix=''):
    return {
        'name': name,
        'passed': bool(passed),
        'risk': risk,
        'detail': detail,
        'suggested_fix': fix,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Audit ACSM extraction/scoring configs for fair evaluation.'
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--score-py', default='wespeaker/bin/score.py')
    parser.add_argument('--trial-list', action='append', default=[])
    parser.add_argument('--embedding-scp')
    parser.add_argument('--output-json')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model = cfg.get('model', '')
    acsm = cfg.get('model_args', {}).get('acsm_args', cfg.get('acsm_args', {}))
    allow_oracle = bool(cfg.get('allow_oracle_age_eval', False))
    oracle_used = bool(cfg.get('oracle_age_used', False))
    checks = []

    checks.append(
        _check('acsm_model_uses_predicted_posterior',
               model in ('ResNet34_ACSM', 'ACSM_ResNet34'),
               'MEDIUM',
               'ACSM forward has no age_group argument and uses predicted age_posterior.'
               if model in ('ResNet34_ACSM', 'ACSM_ResNet34') else
               'Config is not an ACSM model.'))
    checks.append(
        _check('no_extraction_age_group',
               'age_group' not in cfg and 'age_groups' not in cfg,
               'HIGH',
               'Extraction config does not pass age_group.'
               if 'age_group' not in cfg and 'age_groups' not in cfg else
               'Extraction config includes age_group fields.',
               'Remove age_group from extraction config.'))
    checks.append(
        _check('age_label_file_not_extraction_input',
               'age_label_file' not in cfg,
               'MEDIUM',
               'No top-level extraction age_label_file is required. Training acsm_args.age_label_file, if present, is ignored by extract.py dataset construction.'
               if 'age_label_file' not in cfg else
               'Top-level age_label_file is present in extraction config.',
               'Do not pass top-level age labels to extraction.'))
    checks.append(
        _check('oracle_age_explicit',
               (not oracle_used) or allow_oracle,
               'HIGH',
               'Oracle age is disabled or explicitly allowed.',
               'Set allow_oracle_age_eval=true and oracle_age_used=True only for diagnostics.'
               ))
    checks.append(
        _check('oracle_not_default',
               not oracle_used,
               'HIGH',
               'oracle_age_used is false by default.'
               if not oracle_used else 'oracle_age_used=True in config.',
               'Do not report oracle mode as main result.'))

    if os.path.exists(args.score_py):
        with open(args.score_py) as f:
            score_text = f.read()
        reads_age = bool(re.search(r'age(_label|_group|group)', score_text))
        checks.append(
            _check('score_py_no_age_label', not reads_age, 'HIGH',
                   'score.py does not reference age labels.'
                   if not reads_age else
                   'score.py contains age-related tokens.',
                   'Keep score.py cosine-only for fair ACSM evaluation.'))
    else:
        checks.append(
            _check('score_py_exists', False, 'HIGH',
                   'score.py was not found: {}'.format(args.score_py)))

    for trial in args.trial_list:
        exists = os.path.exists(trial)
        checks.append(
            _check('trial_list_exists:{}'.format(trial), exists, 'HIGH',
                   'trial list exists' if exists else 'trial list missing',
                   'Use fixed Vox-CA/MIM trial files.'))

    if args.embedding_scp and os.path.exists(args.embedding_scp):
        bad = []
        with open(args.embedding_scp) as f:
            for idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) != 2:
                    bad.append((idx + 1, line.strip()))
                    if len(bad) >= 5:
                        break
        checks.append(
            _check('embedding_scp_two_columns', not bad, 'MEDIUM',
                   'embedding scp has key/path columns only.'
                   if not bad else 'bad embedding scp rows: {}'.format(bad),
                   'Do not add age_group columns to scoring inputs.'))

    passed = all(item['passed'] for item in checks)
    report = {'status': 'PASS' if passed else 'FAIL', 'checks': checks}
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        with open(args.output_json, 'w') as f:
            f.write(text + '\n')


if __name__ == '__main__':
    main()
