# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tableprint as tp

import torch
import torch.distributed as dist
import torchnet as tnt
from tqdm import tqdm
from wespeaker.dataset.dataset_utils import apply_cmvn, spec_aug


def _show_progress_bar():
    return (not dist.is_available() or not dist.is_initialized()
            or dist.get_rank() == 0)


def run_epoch(dataloader, epoch_iter, model, criterion, optimizer, scheduler,
              margin_scheduler, epoch, logger, scaler, device, configs):
    model.train()
    # By default use average pooling
    loss_meter = tnt.meter.AverageValueMeter()
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
    acsm_meters = {
        'loss_spk': tnt.meter.AverageValueMeter(),
        'loss_age': tnt.meter.AverageValueMeter(),
        'loss_consistency': tnt.meter.AverageValueMeter(),
        'loss_smooth': tnt.meter.AverageValueMeter(),
        'loss_path': tnt.meter.AverageValueMeter(),
        'loss_acsm_total': tnt.meter.AverageValueMeter(),
        'loss_transport': tnt.meter.AverageValueMeter(),
        'loss_transport_cycle': tnt.meter.AverageValueMeter(),
        'loss_transport_identity': tnt.meter.AverageValueMeter(),
        'loss_acstf_total': tnt.meter.AverageValueMeter(),
        'transport_pair_count': tnt.meter.AverageValueMeter(),
        'transport_gate_mean': tnt.meter.AverageValueMeter(),
        'transport_residual_norm_mean': tnt.meter.AverageValueMeter(),
        'transport_cos_pos_mean': tnt.meter.AverageValueMeter(),
        'weighted_consistency': tnt.meter.AverageValueMeter(),
        'gate_mean': tnt.meter.AverageValueMeter(),
        'gate_std': tnt.meter.AverageValueMeter(),
        'uncertainty_mean': tnt.meter.AverageValueMeter(),
        'residual_norm': tnt.meter.AverageValueMeter(),
        'residual_norm_mean': tnt.meter.AverageValueMeter(),
        'cos_raw_can_mean': tnt.meter.AverageValueMeter(),
        'raw_can_cosine_mean': tnt.meter.AverageValueMeter(),
        'l2_raw_can_mean': tnt.meter.AverageValueMeter(),
        'path_valid_pair_count': tnt.meter.AverageValueMeter(),
        'path_nonzero_batch_ratio': tnt.meter.AverageValueMeter(),
    }
    acsm_seen_batches = 0
    acsm_path_nonzero_batches = 0

    frontend_type = configs['dataset_args'].get('frontend', 'fbank')
    progress_bar = tqdm(total=epoch_iter,
                        desc='Epoch {}'.format(epoch),
                        unit='batch',
                        dynamic_ncols=True,
                        disable=not _show_progress_bar())
    try:
        for i, batch in enumerate(dataloader):
            cur_iter = (epoch - 1) * epoch_iter + i
            scheduler.step(cur_iter)
            margin_scheduler.step(cur_iter)

            utts = batch['key']
            targets = batch['label'].long()
            speakers = batch.get('orig_label', targets).long()
            projection = getattr(model.module, 'projection', None)
            if projection is not None and hasattr(projection, 'weight'):
                num_class = projection.weight.size(0)
                invalid = (targets < 0) | (targets >= num_class)
                if invalid.any():
                    bad_idx = invalid.nonzero(as_tuple=False).view(-1)[:10]
                    bad_keys = [utts[j] for j in bad_idx.tolist()]
                    bad_labels = targets[bad_idx].tolist()
                    rank = dist.get_rank() if dist.is_initialized() else 0
                    logger.error(
                        'invalid speaker label before projection: '
                        'rank={}, epoch={}, batch={}, num_class={}, '
                        'label_min={}, label_max={}, bad_keys={}, bad_labels={}'.
                        format(rank, epoch, i + 1, num_class,
                               int(targets.min()), int(targets.max()),
                               bad_keys, bad_labels))
                    raise ValueError('invalid speaker label before projection')
            targets = targets.to(device)  # (B)
            speakers = speakers.to(device)  # (B)
            age_groups = batch.get('age_group', None)
            if age_groups is not None:
                age_groups = age_groups.long()
                module = model.module
                if hasattr(module, 'num_age_groups'):
                    ignore_index = module.ignore_age_index
                    valid = age_groups != ignore_index
                    invalid = valid & ((age_groups < 0) |
                                       (age_groups >= module.num_age_groups))
                    if invalid.any():
                        bad_idx = invalid.nonzero(as_tuple=False).view(-1)[:10]
                        bad_keys = [utts[j] for j in bad_idx.tolist()]
                        bad_ages = age_groups[bad_idx].tolist()
                        rank = dist.get_rank() if dist.is_initialized() else 0
                        logger.error(
                            'invalid age_group before age-conditioned loss: '
                            'rank={}, epoch={}, batch={}, num_age_groups={}, '
                            'age_min={}, age_max={}, bad_keys={}, bad_ages={}'.
                            format(rank, epoch, i + 1, module.num_age_groups,
                                   int(age_groups.min()), int(age_groups.max()),
                                   bad_keys, bad_ages))
                        raise ValueError(
                            'invalid age_group before age-conditioned loss')
                age_groups = age_groups.to(device)
            if frontend_type == 'fbank':
                features = batch['feat']  # (B,T,F)
                features = features.float().to(device)
            else:  # 's3prl'
                wavs = batch['wav']  # (B,1,W)
                wavs = wavs.squeeze(1).float().to(device)  # (B,W)
                wavs_len = torch.LongTensor([wavs.shape[1]]).repeat(
                    wavs.shape[0]).to(device)  # (B)
                with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
                    frontend = getattr(model.module, 'frontend', None)
                    if frontend is None and hasattr(model.module, 'encoder'):
                        frontend = model.module.encoder.frontend
                    features, _ = frontend(wavs, wavs_len)

            with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
                # apply cmvn
                if configs['dataset_args'].get('cmvn', True):
                    features = apply_cmvn(
                        features,
                        **configs['dataset_args'].get('cmvn_args', {}))
                # spec augmentation
                if configs['dataset_args'].get('spec_aug', False):
                    features = spec_aug(
                        features, **configs['dataset_args']['spec_aug_args'])

                model_outputs = model(features)
                if isinstance(model_outputs, dict):
                    embeds = model_outputs['embedding']
                else:
                    embeds = (model_outputs[-1] if isinstance(
                        model_outputs, tuple) else model_outputs)
                logits = model.module.projection(embeds, targets)
                if isinstance(logits, tuple):
                    logits, spk_loss = logits
                else:
                    spk_loss = criterion(logits, targets)
                loss = spk_loss
                extra_losses = {}
                extra_kind = None
                if isinstance(model_outputs, dict) and hasattr(
                        model.module, 'compute_acsm_losses'):
                    if age_groups is None:
                        ignore_index = model.module.ignore_age_index
                        age_groups = targets.new_full(targets.shape,
                                                     ignore_index)
                    extra_losses = model.module.compute_acsm_losses(
                        model_outputs, speakers, age_groups, epoch=epoch)
                    loss = loss + extra_losses['loss_acsm_total']
                    extra_kind = 'ACSM'

            # loss, acc
            loss_meter.add(loss.item())
            acc_meter.add(logits.cpu().detach().numpy(), targets.cpu().numpy())
            acsm_meters['loss_spk'].add(spk_loss.item())
            if extra_kind == 'ACSM':
                acsm_seen_batches += 1
                path_count = float(
                    extra_losses['path_valid_pair_count'].detach().item())
                if path_count > 0.0:
                    acsm_path_nonzero_batches += 1
                acsm_meters['path_nonzero_batch_ratio'].add(
                    1.0 if path_count > 0.0 else 0.0)
                for name, value in extra_losses.items():
                    if name in acsm_meters:
                        acsm_meters[name].add(value.item())

            # updata the model
            optimizer.zero_grad()
            # scaler does nothing here if enable_amp=False
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            progress_bar.update(1)
            postfix = {
                'loss': '{:.4f}'.format(loss_meter.value()[0]),
                'acc': '{:.2f}'.format(acc_meter.value()[0]),
                'lr': '{:.3g}'.format(scheduler.get_lr()),
                'margin': '{:.3g}'.format(margin_scheduler.get_margin()),
            }
            if extra_kind == 'ACSM':
                postfix.update({
                    'spk':
                    '{:.3f}'.format(acsm_meters['loss_spk'].value()[0]),
                    'age':
                    '{:.3f}'.format(acsm_meters['loss_age'].value()[0]),
                    'acsm':
                    '{:.3f}'.format(
                        acsm_meters['loss_acsm_total'].value()[0]),
                    'gate':
                    '{:.3f}'.format(acsm_meters['gate_mean'].value()[0]),
                    'path_pairs':
                    '{:.1f}'.format(
                        acsm_meters['path_valid_pair_count'].value()[0]),
                    'trj':
                    '{:.3f}'.format(
                        acsm_meters['loss_acstf_total'].value()[0]),
                    'tpairs':
                    '{:.1f}'.format(
                        acsm_meters['transport_pair_count'].value()[0]),
                })
            progress_bar.set_postfix(**postfix)

            # log
            if _show_progress_bar() and (
                    i + 1) % configs['log_batch_interval'] == 0:
                logger.info(
                    tp.row((epoch, i + 1, scheduler.get_lr(),
                            margin_scheduler.get_margin()) +
                           (loss_meter.value()[0], acc_meter.value()[0]),
                           width=10,
                           style='grid'))
                if extra_kind == 'ACSM':
                    msg = 'ACSM ' + ', '.join([
                        '{}={:.6f}'.format(k, v.value()[0])
                        for k, v in acsm_meters.items()
                    ])
                    logger.info(msg)

            if (i + 1) == epoch_iter:
                break
    finally:
        progress_bar.close()

    if _show_progress_bar():
        logger.info(
            tp.row((epoch, i + 1, scheduler.get_lr(),
                    margin_scheduler.get_margin()) +
                   (loss_meter.value()[0], acc_meter.value()[0]),
                   width=10,
                   style='grid'))
        if acsm_seen_batches > 0:
            ratio = float(acsm_path_nonzero_batches) / float(acsm_seen_batches)
            logger.info(
                'ACSM_EPOCH path_nonzero_batch_ratio={:.6f}, '
                'path_valid_pair_count={:.6f}, loss_path={:.6f}, '
                'loss_acstf_total={:.6f}, transport_pair_count={:.6f}, '
                'transport_gate_mean={:.6f}, '
                'transport_residual_norm_mean={:.6f}, '
                'transport_cos_pos_mean={:.6f}, '
                'gate_mean={:.6f}, residual_norm_mean={:.6f}, '
                'raw_can_cosine_mean={:.6f}, loss_age={:.6f}, '
                'loss_consistency={:.6f}, weighted_consistency={:.6f}, '
                'loss_smooth={:.6f}'.format(
                    ratio, acsm_meters['path_valid_pair_count'].value()[0],
                    acsm_meters['loss_path'].value()[0],
                    acsm_meters['loss_acstf_total'].value()[0],
                    acsm_meters['transport_pair_count'].value()[0],
                    acsm_meters['transport_gate_mean'].value()[0],
                    acsm_meters['transport_residual_norm_mean'].value()[0],
                    acsm_meters['transport_cos_pos_mean'].value()[0],
                    acsm_meters['gate_mean'].value()[0],
                    acsm_meters['residual_norm_mean'].value()[0],
                    acsm_meters['raw_can_cosine_mean'].value()[0],
                    acsm_meters['loss_age'].value()[0],
                    acsm_meters['loss_consistency'].value()[0],
                    acsm_meters['weighted_consistency'].value()[0],
                    acsm_meters['loss_smooth'].value()[0]))
