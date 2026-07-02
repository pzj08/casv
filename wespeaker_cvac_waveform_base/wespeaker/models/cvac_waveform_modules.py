import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingAgeHead(nn.Module):

    def __init__(self,
                 embedding_dim,
                 num_age_groups=7,
                 age_hidden_dim=256,
                 age_dropout=0.1,
                 ignore_age_index=-1):
        super().__init__()
        self.num_age_groups = num_age_groups
        self.ignore_age_index = ignore_age_index
        self.net = nn.Sequential(nn.LayerNorm(embedding_dim),
                                 nn.Linear(embedding_dim, age_hidden_dim),
                                 nn.SiLU(), nn.Dropout(age_dropout),
                                 nn.Linear(age_hidden_dim, num_age_groups))

    def forward(self, embedding):
        logits = self.net(embedding)
        posterior = torch.softmax(logits, dim=-1)
        entropy = -(posterior * torch.log(posterior.clamp_min(1e-12))).sum(
            dim=-1, keepdim=True)
        uncertainty = entropy / math.log(self.num_age_groups)
        return {
            'age_logits': logits,
            'age_posterior': posterior,
            'age_pred': posterior.argmax(dim=-1),
            'age_uncertainty': uncertainty.clamp(0.0, 1.0),
        }

    def loss(self, age_logits, age_group):
        valid = age_group.ne(self.ignore_age_index)
        if valid.any():
            return F.cross_entropy(age_logits,
                                   age_group.long(),
                                   ignore_index=self.ignore_age_index)
        return age_logits.sum() * 0.0


class WaveformConditionEncoder(nn.Module):

    def __init__(self, embedding_dim, num_age_groups, condition_dim=256):
        super().__init__()
        in_dim = embedding_dim + 3 * num_age_groups + 2
        self.net = nn.Sequential(nn.LayerNorm(in_dim),
                                 nn.Linear(in_dim, condition_dim), nn.SiLU(),
                                 nn.Linear(condition_dim, condition_dim),
                                 nn.SiLU())

    def forward(self, embedding, q_src, q_tgt, u_src=None, u_tgt=None):
        if u_src is None:
            u_src = q_src.new_zeros(q_src.size(0), 1)
        if u_tgt is None:
            u_tgt = q_tgt.new_zeros(q_tgt.size(0), 1)
        cond = torch.cat(
            [embedding, q_src, q_tgt, q_tgt - q_src, u_src, u_tgt], dim=-1)
        return self.net(cond)


class _DilatedFiLMBlock(nn.Module):

    def __init__(self, hidden_channels, condition_dim, kernel_size, dilation):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(hidden_channels,
                              hidden_channels,
                              kernel_size,
                              dilation=dilation,
                              padding=padding)
        groups = 8 if hidden_channels % 8 == 0 else 1
        self.norm = nn.GroupNorm(groups, hidden_channels)
        self.gamma = nn.Linear(condition_dim, hidden_channels)
        self.beta = nn.Linear(condition_dim, hidden_channels)

    def forward(self, x, condition):
        y = self.norm(self.conv(x))
        gamma = self.gamma(condition).unsqueeze(-1)
        beta = self.beta(condition).unsqueeze(-1)
        y = y * (1.0 + gamma) + beta
        y = F.silu(y)
        return x + y


class DilatedWaveformResidualGenerator(nn.Module):

    def __init__(self,
                 condition_dim=256,
                 hidden_channels=128,
                 num_layers=6,
                 kernel_size=7,
                 dilations=None,
                 residual_scale=0.03,
                 gate_max=0.15,
                 gate_init_bias=-3.0):
        super().__init__()
        if dilations is None:
            dilations = [2**i for i in range(num_layers)]
        self.residual_scale = residual_scale
        self.gate_max = gate_max
        self.input = nn.Conv1d(1, hidden_channels, kernel_size=7, padding=3)
        self.blocks = nn.ModuleList([
            _DilatedFiLMBlock(hidden_channels, condition_dim, kernel_size, d)
            for d in dilations[:num_layers]
        ])
        self.output = nn.Conv1d(hidden_channels, 1, kernel_size=7, padding=3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        gate_hidden = max(1, condition_dim // 2)
        self.gate = nn.Sequential(nn.Linear(condition_dim, gate_hidden),
                                  nn.SiLU(),
                                  nn.Linear(gate_hidden, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, gate_init_bias)

    def forward(self, waveform, condition):
        x = waveform
        y = self.input(x.unsqueeze(1))
        for block in self.blocks:
            y = block(y, condition)
        residual = self.output(y).squeeze(1)
        gate = self.gate_max * torch.sigmoid(self.gate(condition))
        x_hat = x + self.residual_scale * gate.view(-1, 1) * residual
        x_hat = torch.clamp(x_hat, -1.0, 1.0)
        return {
            'waveform': x_hat,
            'residual': residual,
            'gate': gate,
            'residual_l1': residual.abs().mean(),
            'residual_rms': torch.sqrt(residual.pow(2).mean() + 1e-12),
        }


class WaveformCounterfactualAgingGenerator(nn.Module):

    def __init__(self,
                 embedding_dim,
                 num_age_groups=7,
                 condition_dim=256,
                 hidden_channels=128,
                 num_layers=6,
                 kernel_size=7,
                 dilations=None,
                 residual_scale=0.03,
                 gate_max=0.15,
                 gate_init_bias=-3.0):
        super().__init__()
        self.condition_encoder = WaveformConditionEncoder(
            embedding_dim, num_age_groups, condition_dim)
        self.residual_generator = DilatedWaveformResidualGenerator(
            condition_dim=condition_dim,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dilations=dilations,
            residual_scale=residual_scale,
            gate_max=gate_max,
            gate_init_bias=gate_init_bias)

    def forward(self, waveform, embedding, q_src, q_tgt, u_src=None, u_tgt=None):
        condition = self.condition_encoder(embedding, q_src, q_tgt, u_src,
                                           u_tgt)
        return self.residual_generator(waveform, condition)


class MultiResolutionSTFTLoss(nn.Module):

    def __init__(self,
                 fft_sizes=(256, 512, 1024),
                 hop_sizes=(64, 128, 256),
                 win_lengths=(256, 512, 1024),
                 eps=1e-12):
        super().__init__()
        self.fft_sizes = list(fft_sizes)
        self.hop_sizes = list(hop_sizes)
        self.win_lengths = list(win_lengths)
        self.eps = eps

    def _mag(self, x, n_fft, hop_length, win_length):
        window = torch.hann_window(win_length, device=x.device, dtype=x.dtype)
        spec = torch.stft(x,
                          n_fft=n_fft,
                          hop_length=hop_length,
                          win_length=win_length,
                          window=window,
                          center=True,
                          return_complex=True)
        return spec.abs()

    def forward(self, pred, target):
        total = pred.new_tensor(0.0)
        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes,
                                   self.win_lengths):
            pred_mag = self._mag(pred, n_fft, hop, win)
            target_mag = self._mag(target, n_fft, hop, win)
            sc = torch.linalg.vector_norm(target_mag - pred_mag) / (
                torch.linalg.vector_norm(target_mag) + self.eps)
            log_mag = F.l1_loss(torch.log(pred_mag + self.eps),
                                torch.log(target_mag + self.eps))
            total = total + sc + log_mag
        return total / len(self.fft_sizes)


class DifferentiableLogMelFrontend(nn.Module):

    def __init__(self,
                 sample_rate=16000,
                 num_mel_bins=80,
                 n_fft=512,
                 hop_length=160,
                 win_length=400,
                 eps=1e-10):
        super().__init__()
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.eps = eps
        mel = self._build_mel_filterbank(sample_rate, n_fft, num_mel_bins)
        self.register_buffer('mel_filterbank', mel, persistent=False)

    @staticmethod
    def _hz_to_mel(freq):
        return 2595.0 * math.log10(1.0 + freq / 700.0)

    @staticmethod
    def _mel_to_hz(mel):
        return 700.0 * (10.0**(mel / 2595.0) - 1.0)

    @classmethod
    def _build_mel_filterbank(cls, sample_rate, n_fft, num_mel_bins):
        min_mel = cls._hz_to_mel(0.0)
        max_mel = cls._hz_to_mel(sample_rate / 2.0)
        mels = torch.linspace(min_mel, max_mel, num_mel_bins + 2)
        hz = torch.tensor([cls._mel_to_hz(float(m)) for m in mels])
        bins = torch.floor((n_fft + 1) * hz / sample_rate).long()
        fb = torch.zeros(num_mel_bins, n_fft // 2 + 1)
        for i in range(num_mel_bins):
            left = int(bins[i])
            center = int(bins[i + 1])
            right = int(bins[i + 2])
            if center > left:
                fb[i, left:center] = torch.linspace(0.0, 1.0, center - left)
            if right > center:
                fb[i, center:right] = torch.linspace(1.0, 0.0,
                                                     right - center)
        return fb

    def forward(self, waveform):
        window = torch.hann_window(self.win_length,
                                   device=waveform.device,
                                   dtype=waveform.dtype)
        spec = torch.stft(waveform,
                          n_fft=self.n_fft,
                          hop_length=self.hop_length,
                          win_length=self.win_length,
                          window=window,
                          center=True,
                          return_complex=True)
        power = spec.abs().pow(2.0).transpose(1, 2)
        fb = self.mel_filterbank.to(device=waveform.device,
                                    dtype=waveform.dtype)
        mel = torch.matmul(power, fb.t()).clamp_min(self.eps)
        return torch.log(mel)


class WaveformCVACLoss(nn.Module):

    def __init__(self,
                 lambda_cf_align=0.003,
                 lambda_id=0.003,
                 lambda_age=0.001,
                 lambda_cycle=0.001,
                 lambda_neg=0.001,
                 lambda_mrstft=0.002,
                 lambda_mel=0.0,
                 lambda_energy=0.0005,
                 lambda_residual=0.0005,
                 min_age_gap=1,
                 max_pairs=32,
                 bidirectional=True,
                 detach_target=True,
                 detach_source_identity=True,
                 neg_margin=0.15,
                 ignore_age_index=-1,
                 stft_fft_sizes=(256, 512, 1024),
                 stft_hop_sizes=(64, 128, 256),
                 stft_win_lengths=(256, 512, 1024),
                 eps=1e-12,
                 **unused):
        super().__init__()
        self.weights = {
            'align': lambda_cf_align,
            'id': lambda_id,
            'age': lambda_age,
            'cycle': lambda_cycle,
            'neg': lambda_neg,
            'mrstft': lambda_mrstft,
            'mel': lambda_mel,
            'energy': lambda_energy,
            'residual': lambda_residual,
        }
        self.min_age_gap = min_age_gap
        self.max_pairs = max_pairs
        self.bidirectional = bidirectional
        self.detach_target = detach_target
        self.detach_source_identity = detach_source_identity
        self.neg_margin = neg_margin
        self.ignore_age_index = ignore_age_index
        self.eps = eps
        self.mrstft = MultiResolutionSTFTLoss(stft_fft_sizes, stft_hop_sizes,
                                              stft_win_lengths, eps)

    def _zero_dict(self, ref):
        zero = ref.sum() * 0.0
        keys = [
            'loss_wavcvac_align', 'loss_wavcvac_id', 'loss_wavcvac_age',
            'loss_wavcvac_cycle', 'loss_wavcvac_neg',
            'loss_wavcvac_mrstft', 'loss_wavcvac_energy',
            'loss_wavcvac_residual', 'loss_wavcvac_total',
            'wavcvac_pair_count', 'wavcvac_gate_mean',
            'wavcvac_residual_l1', 'wavcvac_residual_rms',
            'wavcvac_align_cos_mean', 'wavcvac_id_cos_mean',
            'wavcvac_neg_cos_mean', 'wavcvac_age_loss_mean'
        ]
        return {key: zero for key in keys}

    def _make_pairs(self, speakers, age_group):
        speakers_cpu = speakers.detach().cpu()
        ages_cpu = age_group.detach().cpu()
        pairs = []
        valid = ages_cpu.ne(self.ignore_age_index)
        for i in range(len(ages_cpu)):
            if not valid[i]:
                continue
            for j in range(i + 1, len(ages_cpu)):
                if not valid[j]:
                    continue
                same_spk = int(speakers_cpu[i]) == int(speakers_cpu[j])
                age_gap = abs(int(ages_cpu[i]) - int(ages_cpu[j]))
                if same_spk and age_gap >= self.min_age_gap:
                    pairs.append((i, j))
                    if self.bidirectional:
                        pairs.append((j, i))
        random.shuffle(pairs)
        return pairs[:self.max_pairs]

    @staticmethod
    def _cos(a, b):
        return F.cosine_similarity(a, b, dim=-1)

    def forward(self,
                waveforms,
                embeddings,
                age_posterior,
                age_uncertainty,
                speakers,
                age_group,
                generator,
                embed_fn,
                age_head=None):
        pairs = self._make_pairs(speakers, age_group)
        if len(pairs) == 0:
            return self._zero_dict(embeddings)

        device = embeddings.device
        src_idx = torch.tensor([p[0] for p in pairs], device=device)
        tgt_idx = torch.tensor([p[1] for p in pairs], device=device)
        x_src = waveforms.index_select(0, src_idx)
        z_src = embeddings.index_select(0, src_idx)
        z_tgt = embeddings.index_select(0, tgt_idx)
        q_src = age_posterior.index_select(0, src_idx)
        q_tgt = age_posterior.index_select(0, tgt_idx)
        u_src = age_uncertainty.index_select(0, src_idx)
        u_tgt = age_uncertainty.index_select(0, tgt_idx)
        target_age = age_group.index_select(0, tgt_idx)

        gen_out = generator(x_src, z_src.detach() if self.detach_source_identity
                            else z_src, q_src, q_tgt, u_src, u_tgt)
        x_cf = gen_out['waveform']
        z_cf = embed_fn(x_cf)
        if isinstance(z_cf, tuple):
            z_cf = z_cf[-1]

        z_tgt_cmp = z_tgt.detach() if self.detach_target else z_tgt
        z_src_cmp = z_src.detach() if self.detach_source_identity else z_src
        align_cos = self._cos(z_cf, z_tgt_cmp)
        id_cos = self._cos(z_cf, z_src_cmp)
        loss_align = (1.0 - align_cos).mean()
        loss_id = (1.0 - id_cos).mean()

        if age_head is not None:
            age_logits = age_head(z_cf)['age_logits']
            loss_age = F.cross_entropy(age_logits,
                                       target_age.long(),
                                       ignore_index=self.ignore_age_index)
        else:
            loss_age = embeddings.sum() * 0.0

        cycle_out = generator(x_cf, z_cf, q_tgt, q_src, u_tgt, u_src)
        z_cycle = embed_fn(cycle_out['waveform'])
        if isinstance(z_cycle, tuple):
            z_cycle = z_cycle[-1]
        loss_cycle = (1.0 - self._cos(z_cycle, z_src_cmp)).mean()

        neg_cos_values = []
        for local_idx, (src, _) in enumerate(pairs):
            neg_candidates = torch.nonzero(speakers.ne(speakers[src]),
                                           as_tuple=False).flatten()
            if neg_candidates.numel() > 0:
                neg_idx = neg_candidates[local_idx % neg_candidates.numel()]
                neg_cos_values.append(
                    self._cos(z_cf[local_idx:local_idx + 1],
                              embeddings[neg_idx:neg_idx + 1]).squeeze(0))
        if neg_cos_values:
            neg_cos = torch.stack(neg_cos_values)
            loss_neg = F.relu(self.neg_margin + neg_cos).mean()
            neg_cos_mean = neg_cos.mean()
        else:
            loss_neg = embeddings.sum() * 0.0
            neg_cos_mean = embeddings.sum() * 0.0

        loss_mrstft = self.mrstft(x_cf, x_src)
        rms_cf = torch.sqrt(x_cf.pow(2).mean(dim=-1) + self.eps)
        rms_src = torch.sqrt(x_src.pow(2).mean(dim=-1) + self.eps)
        loss_energy = (rms_cf - rms_src).abs().mean()
        loss_residual = gen_out['residual'].abs().mean()

        weighted = (self.weights['align'] * loss_align +
                    self.weights['id'] * loss_id +
                    self.weights['age'] * loss_age +
                    self.weights['cycle'] * loss_cycle +
                    self.weights['neg'] * loss_neg +
                    self.weights['mrstft'] * loss_mrstft +
                    self.weights['energy'] * loss_energy +
                    self.weights['residual'] * loss_residual)
        pair_count = embeddings.new_tensor(float(len(pairs)))
        return {
            'loss_wavcvac_align': loss_align,
            'loss_wavcvac_id': loss_id,
            'loss_wavcvac_age': loss_age,
            'loss_wavcvac_cycle': loss_cycle,
            'loss_wavcvac_neg': loss_neg,
            'loss_wavcvac_mrstft': loss_mrstft,
            'loss_wavcvac_energy': loss_energy,
            'loss_wavcvac_residual': loss_residual,
            'loss_wavcvac_total': weighted,
            'wavcvac_pair_count': pair_count,
            'wavcvac_gate_mean': gen_out['gate'].mean(),
            'wavcvac_residual_l1': gen_out['residual_l1'],
            'wavcvac_residual_rms': gen_out['residual_rms'],
            'wavcvac_align_cos_mean': align_cos.mean(),
            'wavcvac_id_cos_mean': id_cos.mean(),
            'wavcvac_neg_cos_mean': neg_cos_mean,
            'wavcvac_age_loss_mean': loss_age.detach(),
        }
