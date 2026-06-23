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
    aorc_meters = {
        'loss_spk': tnt.meter.AverageValueMeter(),
        'loss_oam': tnt.meter.AverageValueMeter(),
        'loss_ord': tnt.meter.AverageValueMeter(),
        'loss_proxy': tnt.meter.AverageValueMeter(),
        'loss_dir': tnt.meter.AverageValueMeter(),
        'loss_caa': tnt.meter.AverageValueMeter(),
        'loss_smooth': tnt.meter.AverageValueMeter(),
    }

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
            aorc_speakers = batch.get('orig_label', targets).long()
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
            aorc_speakers = aorc_speakers.to(device)  # (B)
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
                            'invalid age_group before AORC loss: '
                            'rank={}, epoch={}, batch={}, num_age_groups={}, '
                            'age_min={}, age_max={}, bad_keys={}, bad_ages={}'.
                            format(rank, epoch, i + 1, module.num_age_groups,
                                   int(age_groups.min()), int(age_groups.max()),
                                   bad_keys, bad_ages))
                        raise ValueError('invalid age_group before AORC loss')
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

                model_outputs = model(features)  # (embed_a, embed_b) or AORC dict
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
                if isinstance(model_outputs, dict) and hasattr(
                        model.module, 'compute_aorc_losses'):
                    if age_groups is None:
                        ignore_index = model.module.ignore_age_index
                        age_groups = targets.new_full(targets.shape,
                                                     ignore_index)
                    extra_losses = model.module.compute_aorc_losses(
                        model_outputs, aorc_speakers, age_groups)
                    conf = model.module.config
                    loss = (loss +
                            conf['lambda_oam'] * extra_losses['loss_oam'] +
                            conf['lambda_caa'] * extra_losses['loss_caa'] +
                            conf['lambda_smooth'] *
                            extra_losses['loss_smooth'])

            # loss, acc
            loss_meter.add(loss.item())
            acc_meter.add(logits.cpu().detach().numpy(), targets.cpu().numpy())
            aorc_meters['loss_spk'].add(spk_loss.item())
            for name, value in extra_losses.items():
                aorc_meters[name].add(value.item())

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
            if extra_losses:
                postfix.update({
                    'spk': '{:.3f}'.format(aorc_meters['loss_spk'].value()[0]),
                    'oam': '{:.3f}'.format(aorc_meters['loss_oam'].value()[0]),
                    'dir': '{:.3f}'.format(aorc_meters['loss_dir'].value()[0]),
                    'caa': '{:.3f}'.format(aorc_meters['loss_caa'].value()[0]),
                })
                if hasattr(model.module, 'residual_scale_value'):
                    postfix['r'] = '{:.3g}'.format(
                        model.module.residual_scale_value().float().item())
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
                if extra_losses:
                    msg = 'AORC ' + ', '.join([
                        '{}={:.6f}'.format(k, v.value()[0])
                        for k, v in aorc_meters.items()
                    ])
                    if hasattr(model.module, 'residual_scale_value'):
                        msg += ', residual_scale={:.6f}'.format(
                            model.module.residual_scale_value().float().item())
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
