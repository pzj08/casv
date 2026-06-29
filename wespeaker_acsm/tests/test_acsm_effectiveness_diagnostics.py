from types import SimpleNamespace

import numpy as np

import tools.diagnose_acsm_trajectory as traj


def _write_metadata(tmp_path, rows):
    utt2spk = tmp_path / 'utt2spk'
    age_labels = tmp_path / 'age_labels'
    utt2spk.write_text(
        ''.join('{} {}\n'.format(utt, spk) for utt, spk, _ in rows),
        encoding='utf-8')
    age_labels.write_text(
        ''.join('{} {}\n'.format(utt, age) for utt, _, age in rows),
        encoding='utf-8')
    return str(utt2spk), str(age_labels)


def _args(tmp_path,
          rows,
          max_same_pairs=200000,
          max_diff_pairs=200000,
          age_gap_buckets='5,10,15,20'):
    utt2spk, age_label_file = _write_metadata(tmp_path, rows)
    return SimpleNamespace(
        mode='embedding',
        config=None,
        checkpoint=None,
        utt2spk=utt2spk,
        age_label_file=age_label_file,
        age_label_type='value',
        age_bins='21,31,41,51,61,71',
        ignore_age_index=-1,
        raw_embedding_scp=None,
        canonical_embedding_scp=None,
        raw_embeddings=None,
        canonical_embeddings=None,
        wav_scp=None,
        data_list=None,
        data_type='raw',
        utt_list=None,
        trial_file=None,
        diagnostic_json=None,
        train_log=None,
        effectiveness_report=True,
        age_gap_buckets=age_gap_buckets,
        max_utts=None,
        max_pairs=200000,
        max_same_pairs=max_same_pairs,
        max_diff_pairs=max_diff_pairs,
        seed=1234,
        collapse_diff_threshold=0.01,
        identity_cos_threshold=0.9999,
        identity_delta_threshold=1.0e-5,
        bootstrap=0,
        batch_size=4,
        device='cpu',
        save_utterance_diagnostics=None,
        save_embeddings_dir=None,
        output_json=None,
    )


def _report(tmp_path, rows, raw, canonical, **kwargs):
    args = _args(tmp_path, rows, **kwargs)
    report, _, _, _ = traj._build_report(args, raw, canonical, {})
    return report


def test_near_identity_case_flags_risk(tmp_path):
    rows = [
        ('u1', 's1', 20),
        ('u2', 's1', 40),
        ('v1', 's2', 20),
        ('v2', 's2', 40),
    ]
    raw = {
        'u1': np.array([1.0, 0.0, 0.0]),
        'u2': np.array([0.8, 0.6, 0.0]),
        'v1': np.array([0.0, 1.0, 0.0]),
        'v2': np.array([0.0, 0.8, 0.6]),
    }
    canonical = {
        key: value + np.array([1.0e-7, -1.0e-7, 0.0])
        for key, value in raw.items()
    }
    report = _report(tmp_path, rows, raw, canonical)
    decision = report['effectiveness_decision']
    assert decision['near_identity_risk'] is True
    assert decision['embedding_changed'] is False
    assert decision['overall'] in ('FAIL', 'PARTIAL')


def test_effective_cross_age_alignment_passes(tmp_path):
    rows = [
        ('u1', 's1', 20),
        ('u2', 's1', 40),
        ('v1', 's2', 20),
        ('v2', 's2', 40),
    ]
    raw = {
        'u1': np.array([1.0, 0.0, 0.0, 0.0]),
        'u2': np.array([0.8, 0.6, 0.0, 0.0]),
        'v1': np.array([0.0, 0.0, 1.0, 0.0]),
        'v2': np.array([0.0, 0.0, 0.8, 0.6]),
    }
    canonical = {
        'u1': np.array([1.0, 0.0, 0.0, 0.0]),
        'u2': np.array([0.98, 0.2, 0.0, 0.0]),
        'v1': np.array([0.0, 0.0, 1.0, 0.0]),
        'v2': np.array([0.0, 0.0, 0.98, 0.2]),
    }
    report = _report(tmp_path, rows, raw, canonical)
    decision = report['effectiveness_decision']
    assert decision['embedding_changed'] is True
    assert decision['same_cross_age_improved'] is True
    assert decision['different_speaker_preserved'] is True
    assert decision['collapse_risk'] is False
    assert decision['overall'] == 'PASS'


def test_collapse_case_fails_even_if_same_speaker_improves(tmp_path):
    rows = [
        ('u1', 's1', 20),
        ('u2', 's1', 40),
        ('v1', 's2', 20),
        ('v2', 's2', 40),
    ]
    raw = {
        'u1': np.array([1.0, 0.0, 0.0]),
        'u2': np.array([0.8, 0.6, 0.0]),
        'v1': np.array([0.0, 1.0, 0.0]),
        'v2': np.array([0.0, 0.8, 0.6]),
    }
    canonical = {
        'u1': np.array([1.0, 0.0, 0.0]),
        'u2': np.array([1.0, 0.0, 0.0]),
        'v1': np.array([1.0, 0.0, 0.0]),
        'v2': np.array([1.0, 0.0, 0.0]),
    }
    report = _report(tmp_path, rows, raw, canonical)
    decision = report['effectiveness_decision']
    assert decision['same_cross_age_improved'] is True
    assert decision['collapse_risk'] is True
    assert decision['overall'] == 'FAIL'


def test_insufficient_pairs_case_reports_warning(tmp_path):
    rows = [
        ('u1', 's1', 20),
        ('u2', 's1', 20),
        ('v1', 's2', 40),
    ]
    raw = {
        'u1': np.array([1.0, 0.0]),
        'u2': np.array([0.9, 0.1]),
        'v1': np.array([0.0, 1.0]),
    }
    canonical = {key: value.copy() for key, value in raw.items()}
    report = _report(tmp_path, rows, raw, canonical)
    assert report['sampled_pair_counts']['num_same_cross_age_pairs'] == 0
    assert report['same_speaker_cross_age']['num_pairs'] == 0
    assert report['effectiveness_decision']['same_cross_age_improved'] is None
    assert report['effectiveness_decision']['overall'] in ('FAIL',
                                                           'INSUFFICIENT')
    assert any('same_speaker_cross_age' in warning
               for warning in report['warnings'])


def test_age_gap_bucket_counts(tmp_path):
    rows = [
        ('u20', 's1', 20),
        ('u26', 's1', 26),
        ('u31', 's1', 31),
        ('u41', 's1', 41),
    ]
    raw = {
        'u20': np.array([1.0, 0.0]),
        'u26': np.array([0.9, 0.1]),
        'u31': np.array([0.8, 0.2]),
        'u41': np.array([0.7, 0.3]),
    }
    canonical = {
        'u20': np.array([1.0, 0.0]),
        'u26': np.array([0.95, 0.05]),
        'u31': np.array([0.9, 0.1]),
        'u41': np.array([0.85, 0.15]),
    }
    report = _report(tmp_path, rows, raw, canonical)
    buckets = report['age_gap_buckets']
    assert buckets['gap_ge_5']['num_pairs'] == 6
    assert buckets['gap_ge_10']['num_pairs'] == 4
    assert buckets['gap_ge_15']['num_pairs'] == 2
    assert buckets['gap_ge_20']['num_pairs'] == 1
