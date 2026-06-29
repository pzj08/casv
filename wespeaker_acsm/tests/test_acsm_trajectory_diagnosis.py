import json
from types import SimpleNamespace

import kaldiio
import numpy as np
import pytest
import torch
import yaml

import tools.diagnose_acsm_trajectory as traj


def _write_vec_scp(tmp_path, name, vectors):
    ark = tmp_path / f'{name}.ark'
    scp = tmp_path / f'{name}.scp'
    with kaldiio.WriteHelper(f'ark,scp:{ark},{scp}') as writer:
        for key, value in vectors.items():
            writer(key, np.asarray(value, dtype=np.float32))
    return str(scp)


def _write_metadata(tmp_path):
    utt2spk = tmp_path / 'utt2spk'
    utt2spk.write_text(
        'utt1 spk1\nutt2 spk1\nutt3 spk2\nutt4 spk2\n',
        encoding='utf-8')
    age_path = tmp_path / 'age.npy'
    np.save(str(age_path), {
        'utt1': 25,
        'utt2': 45,
        'utt3': 35,
        'utt4': 35,
    })
    return str(utt2spk), str(age_path)


def _base_args(tmp_path, raw_scp, can_scp, diagnostic_json=None):
    utt2spk, age_path = _write_metadata(tmp_path)
    return SimpleNamespace(
        mode='embedding',
        config=None,
        checkpoint=None,
        utt2spk=utt2spk,
        age_label_file=age_path,
        age_label_type='value',
        age_bins='21,31,41,51,61,71',
        ignore_age_index=-1,
        raw_embedding_scp=raw_scp,
        canonical_embedding_scp=can_scp,
        wav_scp=None,
        data_list=None,
        data_type='raw',
        utt_list=None,
        trial_file=None,
        diagnostic_json=diagnostic_json,
        train_log=None,
        max_utts=None,
        max_pairs=2,
        batch_size=2,
        device='cpu',
        save_utterance_diagnostics=None,
        save_embeddings_dir=None,
        output_json=None,
    )


def test_embedding_mode_marks_internal_metrics_unavailable(tmp_path):
    raw_scp = _write_vec_scp(tmp_path, 'raw', {
        'utt1': [1.0, 0.0],
        'utt2': [0.8, 0.2],
        'utt3': [0.0, 1.0],
        'utt4': [0.2, 0.8],
    })
    can_scp = _write_vec_scp(tmp_path, 'can', {
        'utt1': [1.0, 0.0],
        'utt2': [0.95, 0.05],
        'utt3': [0.0, 1.0],
        'utt4': [0.2, 0.8],
    })
    args = _base_args(tmp_path, raw_scp, can_scp)
    raw = traj._read_embeddings(raw_scp)
    can = traj._read_embeddings(can_scp)
    report, _, _, _ = traj._build_report(args, raw, can, {})
    assert report['oracle_age_used'] is False
    assert report['gate_mean'] is None
    assert 'gate_mean' in report['unavailable_metrics']
    assert report['raw_can_cosine_mean'] is not None
    assert report['same_speaker_cross_age_distance_raw'] is not None


def test_embedding_mode_uses_diagnostic_json(tmp_path):
    raw_scp = _write_vec_scp(tmp_path, 'raw', {
        'utt1': [1.0, 0.0],
        'utt2': [0.8, 0.2],
    })
    can_scp = _write_vec_scp(tmp_path, 'can', {
        'utt1': [1.0, 0.0],
        'utt2': [0.95, 0.05],
    })
    diag = tmp_path / 'diag.json'
    diag.write_text(json.dumps({
        'gate_mean': 0.08,
        'gate_std': 0.01,
        'residual_norm_mean': 0.13,
    }), encoding='utf-8')
    args = _base_args(tmp_path, raw_scp, can_scp, str(diag))
    report, _, _, _ = traj._build_report(args, traj._read_embeddings(raw_scp),
                                         traj._read_embeddings(can_scp), {})
    assert report['gate_mean'] == 0.08
    assert report['residual_norm_mean'] == 0.13


def test_model_mode_extracts_internal_state_without_age_forward(tmp_path,
                                                               monkeypatch):
    utt2spk, age_path = _write_metadata(tmp_path)
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({
        'model': 'ResNet34_ACSM',
        'model_args': {
            'feat_dim': 2,
            'embed_dim': 2,
            'acsm_args': {
                'enabled': True,
                'num_age_groups': 4,
                'age_bins': [21, 31, 41],
                'age_label_file': None,
            },
        },
        'dataset_args': {
            'frontend': 'fbank',
            'fbank_args': {
                'num_mel_bins': 2,
                'frame_shift': 10,
                'frame_length': 25,
                'dither': 0.0,
            },
            'cmvn': False,
        },
    }), encoding='utf-8')

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.called_args = None

        def forward(self, x):
            self.called_args = x
            b = x.shape[0]
            raw = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0],
                                [0.2, 0.8]])[:b]
            can = torch.tensor([[1.0, 0.0], [0.95, 0.05], [0.0, 1.0],
                                [0.2, 0.8]])[:b]
            return {
                'raw_embedding': raw,
                'embedding': can,
                'age_posterior': torch.full((b, 4), 0.25),
                'gate': torch.full((b, 1), 0.08),
                'uncertainty': torch.full((b,), 0.75),
                'canonical_residual': can - raw,
                'age_pred': torch.arange(b) % 4,
            }

    fake_model = FakeModel()
    monkeypatch.setattr(traj, 'get_speaker_model',
                        lambda name: (lambda **kwargs: fake_model))
    monkeypatch.setattr(traj, 'load_checkpoint', lambda *a, **k: None)
    monkeypatch.setattr(
        traj, 'Dataset',
        lambda *a, **k: [
            {'key': 'utt1', 'feat': torch.ones(3, 2)},
            {'key': 'utt2', 'feat': torch.ones(4, 2)},
            {'key': 'utt3', 'feat': torch.ones(3, 2)},
            {'key': 'utt4', 'feat': torch.ones(4, 2)},
        ])
    args = SimpleNamespace(
        mode='model',
        config=str(config),
        checkpoint=str(tmp_path / 'model.pt'),
        utt2spk=utt2spk,
        age_label_file=age_path,
        age_label_type='value',
        age_bins='21,31,41,51,61,71',
        ignore_age_index=-1,
        raw_embedding_scp=None,
        canonical_embedding_scp=None,
        wav_scp=None,
        data_list=str(tmp_path / 'unused.jsonl'),
        data_type='raw',
        utt_list=None,
        trial_file=None,
        diagnostic_json=None,
        train_log=None,
        max_utts=None,
        max_pairs=1,
        batch_size=4,
        device='cpu',
        save_utterance_diagnostics=None,
        save_embeddings_dir=None,
        output_json=None,
    )
    raw, can, utt_diag = traj._extract_model_mode(args)
    assert set(raw) == {'utt1', 'utt2', 'utt3', 'utt4'}
    assert fake_model.called_args is not None
    assert len(utt_diag) == 4
    assert utt_diag['utt1']['gate_mean'] == pytest.approx(0.08)

    report, valid_keys, _, ages = traj._build_report(args, raw, can, utt_diag)
    assert report['oracle_age_used'] is False
    assert report['gate_mean'] == pytest.approx(0.08)
    assert report['uncertainty_mean'] == pytest.approx(0.75)
    assert report['residual_norm_mean'] is not None
    assert report['same_speaker_cross_age_distance_raw'] is not None
    assert report['same_speaker_cross_age_pair_count'] <= 1
    assert ages['utt1'] != -1
    assert valid_keys


def test_utterance_diagnostics_and_embedding_save(tmp_path):
    utt2spk, age_path = _write_metadata(tmp_path)
    keys = ['utt1', 'utt2']
    raw = {'utt1': np.array([1.0, 0.0]), 'utt2': np.array([0.8, 0.2])}
    can = {'utt1': np.array([1.0, 0.0]), 'utt2': np.array([0.95, 0.05])}
    diag = {
        'utt1': {'age_pred': 1, 'age_entropy': 0.7, 'gate_mean': 0.08,
                 'residual_norm': 0.0, 'raw_can_cosine': 1.0},
        'utt2': {'age_pred': 2, 'age_entropy': 0.8, 'gate_mean': 0.09,
                 'residual_norm': 0.2, 'raw_can_cosine': 0.98},
    }
    ages = traj._read_age_labels(age_path, keys, 'value',
                                 [21, 31, 41, 51, 61, 71], -1)
    out_jsonl = tmp_path / 'utt_diag.jsonl'
    traj._save_utterance_diagnostics(str(out_jsonl), keys,
                                     traj._read_utt2spk(utt2spk), ages, diag)
    rows = [json.loads(x) for x in out_jsonl.read_text().splitlines()]
    assert rows[0]['age_group'] is not None
    assert rows[0]['age_pred'] == 1

    emb_dir = tmp_path / 'emb'
    traj._save_embeddings(str(emb_dir), keys, raw, can, diag)
    assert (emb_dir / 'raw_embeddings.npy').exists()
    assert (emb_dir / 'canonical_embeddings.npy').exists()
    assert (emb_dir / 'utts.txt').exists()
