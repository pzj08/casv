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

import os
import re
import math
import warnings
import subprocess
from pprint import pformat

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore',
                        message='pkg_resources is deprecated as an API.*',
                        category=UserWarning)
warnings.filterwarnings('ignore',
                        message='torchaudio._backend.set_audio_backend.*',
                        category=UserWarning)

import fire
import numpy as np
import tableprint as tp
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader

import wespeaker.utils.schedulers as schedulers
from wespeaker.dataset.dataset import Dataset
from wespeaker.models.acsm_modules import acsm_is_enabled, get_acsm_config
from wespeaker.models.projections import get_projection
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.checkpoint import load_checkpoint, save_checkpoint
from wespeaker.utils.executor import run_epoch
from wespeaker.utils.file_utils import read_table
from wespeaker.utils.utils import get_logger, parse_config_or_kwargs, set_seed, \
    spk2id


def _age_value_to_group(age_value, age_bins):
    age_value = float(age_value)
    for idx, boundary in enumerate(age_bins):
        if age_value < float(boundary):
            return idx
    return len(age_bins)


def _is_ignore_age_value(value, ignore_index):
    if isinstance(value, str):
        value = value.strip()
        if value.lower() == 'nan':
            return True
        if value == str(ignore_index):
            return True
    try:
        value = float(value)
        return math.isnan(value) or value == float(ignore_index)
    except (TypeError, ValueError):
        return False


def _load_age_labels(age_label_file, age_label_type, age_bins,
                     num_age_groups, ignore_index):
    age_labels = {}
    if age_label_file.endswith('.npy'):
        loaded = np.load(age_label_file, allow_pickle=True)
        if getattr(loaded, 'shape', None) == ():
            loaded = loaded.item()
        if not isinstance(loaded, dict):
            raise ValueError('npy age_label_file must contain a dict')
        rows = loaded.items()
    else:
        rows = ((row[0], row[1]) for row in read_table(age_label_file)
                if len(row) >= 2)
    for key, value in rows:
        if _is_ignore_age_value(value, ignore_index):
            age_group = ignore_index
        elif age_label_type == 'value':
            if age_bins is None:
                raise ValueError(
                    'age_label_type=value requires age_bins in config')
            age_group = _age_value_to_group(value, age_bins)
        else:
            age_group = int(value)
        if age_group != ignore_index and not (0 <= age_group < num_age_groups):
            raise ValueError(
                'age group {} for {} is outside [0, {})'.format(
                    age_group, key, num_age_groups))
        age_labels[key] = age_group
    return age_labels


def _acsm_needs_age_labels(acsm_conf):
    losses = acsm_conf.get('losses', {})
    return (float(losses.get('lambda_age', 0.0)) > 0.0
            or float(losses.get('lambda_path', 0.0)) > 0.0)


def _validate_acsm_age_label_config(acsm_conf):
    if _acsm_needs_age_labels(acsm_conf) and acsm_conf[
            'age_label_file'] is None:
        raise ValueError(
            'ACSM lambda_age/lambda_path is enabled but age_label_file is missing'
        )


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ('true', '1', 'yes', 'y')
    return bool(value)


def _git_commit_hash():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def _experiment_manifest(configs, acsm_conf=None):
    manifest = {
        'git_commit': _git_commit_hash(),
        'seed': configs.get('seed'),
        'model': configs.get('model'),
        'model_init': configs.get('model_init'),
        'checkpoint': configs.get('checkpoint'),
        'train_data': configs.get('train_data'),
        'train_label': configs.get('train_label'),
        'key_filter_file': configs.get('key_filter_file'),
        'exp_dir': configs.get('exp_dir'),
        'score_path': None,
        'metric_path': None,
    }
    if acsm_conf is not None:
        losses = acsm_conf.get('losses', {})
        manifest['acsm'] = {
            'age_label_file':
            acsm_conf.get('age_label_file'),
            'num_age_groups':
            acsm_conf.get('num_age_groups'),
            'reference_age_group':
            acsm_conf.get('reference_age_group'),
            'lambda_age':
            losses.get('lambda_age'),
            'lambda_consistency':
            losses.get('lambda_consistency'),
            'lambda_smooth':
            losses.get('lambda_smooth'),
            'lambda_path':
            losses.get('lambda_path'),
        }
    return manifest


def train(config='conf/config.yaml', **kwargs):
    """Trains a model on the given features and spk labels.

    :config: A training configuration. Note that all parameters in the
             config can also be manually adjusted with --ARG VALUE
    :returns: None
    """
    configs = parse_config_or_kwargs(config, **kwargs)
    checkpoint = configs.get('checkpoint', None)
    acsm_conf = get_acsm_config(configs)
    use_acsm = acsm_is_enabled(configs)
    if use_acsm:
        configs.setdefault('model_args', {})
        configs['model_args']['acsm_args'] = acsm_conf
    # dist configs
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ['WORLD_SIZE'])
    gpu = int(configs['gpus'][local_rank])
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl')

    train_conf = configs.get('train', {})
    target_global_batch = train_conf.get('effective_global_batch_size', None)
    if target_global_batch is not None:
        if target_global_batch % world_size != 0:
            raise ValueError(
                'effective_global_batch_size={} is not divisible by world_size={}'.format(
                    target_global_batch, world_size))
        per_gpu_batch = target_global_batch // world_size
        if train_conf.get('auto_scale_per_gpu_batch_size', False):
            configs['dataloader_args']['batch_size'] = per_gpu_batch
        effective_global = configs['dataloader_args']['batch_size'] * world_size
        if effective_global != target_global_batch:
            raise ValueError(
                'per-GPU batch_size={} with world_size={} gives global batch {}, expected {}'.format(
                    configs['dataloader_args']['batch_size'], world_size,
                    effective_global, target_global_batch))

    model_dir = os.path.join(configs['exp_dir'], "models")
    if rank == 0:
        try:
            os.makedirs(model_dir)
        except IOError:
            print("[warning] " + model_dir + " already exists !!!")
            if checkpoint is None:
                print("[error] checkpoint is null !")
                exit(1)
    dist.barrier(device_ids=[gpu])  # let the rank 0 mkdir first

    logger = get_logger(configs['exp_dir'], 'train.log')
    if world_size > 1:
        logger.info('training on multiple gpus, this gpu {}'.format(gpu))

    if rank == 0:
        logger.info("exp_dir is: {}".format(configs['exp_dir']))
        logger.info("world_size: {}, per_gpu_batch_size: {}, effective_global_batch_size: {}".format(
            world_size, configs['dataloader_args']['batch_size'],
            world_size * configs['dataloader_args']['batch_size']))
        logger.info("<== Passed Arguments ==>")
        # Print arguments into logs
        for line in pformat(configs).split('\n'):
            logger.info(line)

    # seed
    set_seed(configs['seed'] + rank)

    # train data
    train_label = configs['train_label']
    train_utt_spk_list = read_table(train_label)
    spk2id_dict = spk2id(train_utt_spk_list)
    age_labels = None
    age_ignore_index = acsm_conf['ignore_age_index']
    if use_acsm:
        _validate_acsm_age_label_config(acsm_conf)
        if acsm_conf['age_label_file'] is not None:
            age_labels = _load_age_labels(acsm_conf['age_label_file'],
                                          acsm_conf['age_label_type'],
                                          acsm_conf['age_bins'],
                                          acsm_conf['num_age_groups'],
                                          acsm_conf['ignore_age_index'])
    if rank == 0:
        logger.info("<== Data statistics ==>")
        logger.info("train data num: {}, spk num: {}".format(
            len(train_utt_spk_list), len(spk2id_dict)))
        if age_labels is not None:
            logger.info("age label num: {}, num_age_groups: {}".format(
                len(age_labels), acsm_conf['num_age_groups']))

    # dataset and dataloader
    train_dataset = Dataset(configs['data_type'],
                            configs['train_data'],
                            configs['dataset_args'],
                            spk2id_dict,
                            reverb_lmdb_file=configs.get('reverb_data', None),
                            noise_lmdb_file=configs.get('noise_data', None),
                            key_filter_file=configs.get('key_filter_file', None),
                            age_labels=age_labels,
                            ignore_age_index=age_ignore_index)
    train_dataloader = DataLoader(train_dataset, **configs['dataloader_args'])
    batch_size = configs['dataloader_args']['batch_size']
    if configs['dataset_args'].get('sample_num_per_epoch', 0) > 0:
        sample_num_per_epoch = configs['dataset_args']['sample_num_per_epoch']
    else:
        sample_num_per_epoch = len(train_utt_spk_list)
    epoch_iter = sample_num_per_epoch // world_size // batch_size
    if rank == 0:
        logger.info("<== Dataloaders ==>")
        logger.info("train dataloaders created")
        logger.info('epoch iteration number: {}'.format(epoch_iter))

    # model: frontend (optional) => speaker model => projection layer
    logger.info("<== Model ==>")
    frontend_type = configs['dataset_args'].get('frontend', 'fbank')
    if frontend_type != "fbank":
        from wespeaker.frontend import frontend_class_dict
        frontend_args = frontend_type + "_args"
        frontend = frontend_class_dict[frontend_type](
            **configs['dataset_args'][frontend_args],
            sample_rate=configs['dataset_args']['resample_rate'])
        configs['model_args']['feat_dim'] = frontend.output_size()
        model = get_speaker_model(configs['model'])(**configs['model_args'])
        model.add_module("frontend", frontend)
    else:
        model = get_speaker_model(configs['model'])(**configs['model_args'])
    if rank == 0:
        num_params = sum(param.numel() for param in model.parameters())
        logger.info('speaker_model size: {}'.format(num_params))
    # For model_init, only frontend and speaker model are needed !!!
    if configs['model_init'] is not None:
        logger.info('Load initial model from {}'.format(configs['model_init']))
        model_init_strict = _as_bool(configs.get('model_init_strict', False))
        load_checkpoint(model,
                        configs['model_init'],
                        strict=model_init_strict,
                        allow_acsm_partial=use_acsm
                        and not model_init_strict,
                        logger=logger)
    elif checkpoint is None:
        logger.info('Train model from scratch ...')
    if use_acsm and rank == 0:
        logger.info('ACSM enabled: {}'.format(acsm_conf))
    # projection layer
    configs['projection_args']['embed_dim'] = configs['model_args'][
        'embed_dim']
    configs['projection_args']['num_class'] = len(spk2id_dict)
    configs['projection_args']['do_lm'] = configs.get('do_lm', False)
    if configs['data_type'] != 'feat' and configs['dataset_args'][
            'speed_perturb']:
        # diff speed is regarded as diff spk
        configs['projection_args']['num_class'] *= 3
        if configs.get('do_lm', False):
            logger.info(
                'No speed perturb while doing large margin fine-tuning')
            configs['dataset_args']['speed_perturb'] = False
    projection = get_projection(configs['projection_args'])
    model.add_module("projection", projection)
    if rank == 0:
        # print model
        for line in pformat(model).split('\n'):
            logger.info(line)
        # !!!IMPORTANT!!!
        # Try to export the model by script, if fails, we should refine
        # the code to satisfy the script export requirements
        if frontend_type == 'fbank' and not use_acsm:
            script_model = torch.jit.script(model)
            script_model.save(os.path.join(model_dir, 'init.zip'))

    # If specify checkpoint, load some info from checkpoint.
    # For checkpoint, frontend, speaker model, and projection layer
    # are all needed !!!
    if checkpoint is not None:
        load_checkpoint(model, checkpoint)
        start_epoch = int(re.findall(r"(?<=model_)\d*(?=.pt)",
                                     checkpoint)[0]) + 1
        logger.info('Load checkpoint: {}'.format(checkpoint))
    else:
        start_epoch = 1
    logger.info('start_epoch: {}'.format(start_epoch))

    # ddp_model
    model.cuda()
    ddp_model = torch.nn.parallel.DistributedDataParallel(model)
    device = torch.device("cuda")

    criterion = getattr(torch.nn, configs['loss'])(**configs['loss_args'])
    if rank == 0:
        logger.info("<== Loss ==>")
        logger.info("loss criterion is: " + configs['loss'])

    if 'initial_lr' in configs['scheduler_args']:
        configs['optimizer_args']['lr'] = (
            configs['scheduler_args']['initial_lr']
        )
    optimizer = getattr(torch.optim,
                        configs['optimizer'])(ddp_model.parameters(),
                                              **configs['optimizer_args'])
    if rank == 0:
        logger.info("<== Optimizer ==>")
        logger.info("optimizer is: " + configs['optimizer'])

    # scheduler
    configs['scheduler_args']['num_epochs'] = configs['num_epochs']
    configs['scheduler_args']['epoch_iter'] = epoch_iter
    # here, we consider the batch_size 64 as the base, the learning rate will be
    # adjusted according to the batchsize and world_size used in different setup
    configs['scheduler_args']['scale_ratio'] = 1.0 * world_size * configs[
        'dataloader_args']['batch_size'] / 64
    scheduler = getattr(schedulers,
                        configs['scheduler'])(optimizer,
                                              **configs['scheduler_args'])
    if rank == 0:
        logger.info("<== Scheduler ==>")
        logger.info("scheduler is: " + configs['scheduler'])

    # margin scheduler
    configs['margin_update']['epoch_iter'] = epoch_iter
    margin_scheduler = getattr(schedulers, configs['margin_scheduler'])(
        model=model, **configs['margin_update'])
    if rank == 0:
        logger.info("<== MarginScheduler ==>")

    # save config.yaml
    if rank == 0:
        saved_config_path = os.path.join(configs['exp_dir'], 'config.yaml')
        with open(saved_config_path, 'w') as fout:
            data = yaml.dump(configs)
            fout.write(data)
        manifest_path = os.path.join(configs['exp_dir'],
                                     'experiment_manifest.yaml')
        with open(manifest_path, 'w') as fout:
            yaml.safe_dump(_experiment_manifest(
                configs, acsm_conf if use_acsm else None),
                           fout,
                           sort_keys=False)

    # training
    dist.barrier(device_ids=[gpu])  # synchronize here
    if rank == 0:
        logger.info("<========== Training process ==========>")
        header = ['Epoch', 'Batch', 'Lr', 'Margin', 'Loss', "Acc"]
        for line in tp.header(header, width=10, style='grid').split('\n'):
            logger.info(line)
    dist.barrier(device_ids=[gpu])  # synchronize here

    scaler = torch.cuda.amp.GradScaler(enabled=configs['enable_amp'])
    for epoch in range(start_epoch, configs['num_epochs'] + 1):
        train_dataset.set_epoch(epoch)

        run_epoch(train_dataloader,
                  epoch_iter,
                  ddp_model,
                  criterion,
                  optimizer,
                  scheduler,
                  margin_scheduler,
                  epoch,
                  logger,
                  scaler,
                  device=device,
                  configs=configs)

        if rank == 0:
            if epoch % configs['save_epoch_interval'] == 0 or epoch > configs[
                    'num_epochs'] - configs['num_avg']:
                save_checkpoint(
                    model, os.path.join(model_dir,
                                        'model_{}.pt'.format(epoch)))

    if rank == 0:
        os.symlink('model_{}.pt'.format(configs['num_epochs']),
                   os.path.join(model_dir, 'final_model.pt'))
        logger.info(tp.bottom(len(header), width=10, style='grid'))
    dist.destroy_process_group()


if __name__ == '__main__':
    fire.Fire(train)
