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
    cvac_meter_names = [
        'loss_age_head', 'loss_wavcvac_total', 'loss_wavcvac_align',
        'loss_wavcvac_id', 'loss_wavcvac_age', 'loss_wavcvac_cycle',
        'loss_wavcvac_neg', 'loss_wavcvac_mrstft', 'loss_wavcvac_energy',
        'loss_wavcvac_residual', 'wavcvac_pair_count',
        'wavcvac_gate_mean', 'wavcvac_residual_l1',
        'wavcvac_residual_rms', 'wavcvac_align_cos_mean',
        'wavcvac_id_cos_mean', 'wavcvac_neg_cos_mean',
        'wavcvac_age_loss_mean', 'wavcvac_missing_waveform',
        'wavcvac_missing_age'
    ]
    cvac_meters = {
        name: tnt.meter.AverageValueMeter()
        for name in cvac_meter_names
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
            targets = batch['label']
            targets = targets.long().to(device)  # (B)
            if frontend_type == 'fbank':
                features = batch['feat']  # (B,T,F)
                features = features.float().to(device)
            else:  # 's3prl'
                wavs = batch['wav']  # (B,1,W)
                wavs = wavs.squeeze(1).float().to(device)  # (B,W)
                wavs_len = torch.LongTensor([wavs.shape[1]]).repeat(
                    wavs.shape[0]).to(device)  # (B)
                with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
                    features, _ = model.module.frontend(wavs, wavs_len)

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

                outputs = model(features)  # (embed_a,embed_b) in most cases
                embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
                outputs = model.module.projection(embeds, targets)
                if isinstance(outputs, tuple):
                    outputs, loss = outputs
                else:
                    loss = criterion(outputs, targets)

                cvac_values = {
                    name: loss.detach() * 0.0
                    for name in cvac_meter_names
                }
                if getattr(model.module, 'cvac_enabled', False):
                    waveform_key = configs['dataset_args'].get(
                        'waveform_key', 'waveform')
                    has_waveform = waveform_key in batch
                    has_age = 'age_group' in batch
                    cvac_values['wavcvac_missing_waveform'] = loss.new_tensor(
                        0.0 if has_waveform else 1.0)
                    cvac_values['wavcvac_missing_age'] = loss.new_tensor(
                        0.0 if has_age else 1.0)
                    if has_age:
                        age_group = batch['age_group'].long().to(device)
                        age_out = model.module.cvac_age_head(embeds)
                        loss_age_head = model.module.cvac_age_head.loss(
                            age_out['age_logits'], age_group)
                        cvac_values['loss_age_head'] = loss_age_head
                        loss = loss + configs.get('cvac_args', {}).get(
                            'lambda_age_head', 0.0) * loss_age_head
                        if has_waveform:
                            waveforms = batch[waveform_key].float().to(device)

                            def embed_fn(wav):
                                return model.module.forward_waveform_for_cvac(
                                    wav)

                            cvac_out = model.module.cvac_loss(
                                waveforms, embeds, age_out['age_posterior'],
                                age_out['age_uncertainty'], targets,
                                age_group, model.module.cvac_generator,
                                embed_fn, model.module.cvac_age_head)
                            cvac_values.update(cvac_out)
                            loss = loss + cvac_out['loss_wavcvac_total']

            # loss, acc
            loss_meter.add(loss.item())
            for name, value in cvac_values.items():
                cvac_meters[name].add(float(value.detach().cpu()))
            acc_meter.add(outputs.cpu().detach().numpy(), targets.cpu().numpy())

            # updata the model
            optimizer.zero_grad()
            # scaler does nothing here if enable_amp=False
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            progress_bar.update(1)
            progress_bar.set_postfix(
                loss='{:.4f}'.format(loss_meter.value()[0]),
                acc='{:.2f}'.format(acc_meter.value()[0]),
                lr='{:.3g}'.format(scheduler.get_lr()),
                margin='{:.3g}'.format(margin_scheduler.get_margin()),
                cvac='{:.3g}'.format(
                    cvac_meters['loss_wavcvac_total'].value()[0]),
                pairs='{:.1f}'.format(
                    cvac_meters['wavcvac_pair_count'].value()[0]))

            # log
            if _show_progress_bar() and (
                    i + 1) % configs['log_batch_interval'] == 0:
                logger.info(
                    tp.row((epoch, i + 1, scheduler.get_lr(),
                            margin_scheduler.get_margin()) +
                           (loss_meter.value()[0], acc_meter.value()[0]),
                           width=10,
                           style='grid'))
                if getattr(model.module, 'cvac_enabled', False):
                    logger.info(
                        'cvac loss={:.6f} age_head={:.6f} pairs={:.2f} '
                        'missing_waveform={:.2f} missing_age={:.2f}'.format(
                            cvac_meters['loss_wavcvac_total'].value()[0],
                            cvac_meters['loss_age_head'].value()[0],
                            cvac_meters['wavcvac_pair_count'].value()[0],
                            cvac_meters['wavcvac_missing_waveform'].value()[0],
                            cvac_meters['wavcvac_missing_age'].value()[0]))

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
