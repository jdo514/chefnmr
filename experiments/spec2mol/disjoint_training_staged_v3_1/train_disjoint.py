#!/usr/bin/env python3
"""Train one resumable ChefNMR-S segment using the archived augmented config.

Does not alter src.main or the neural architecture. Requires NMRbox training venv.
"""
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import OmegaConf

from src.data.datamodule import NMRDataModule
from src.model.model import NMRTo3DStructureElucidation


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(2 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class FreshPopulationCheckpoint(ModelCheckpoint):
    """Do not import the old strict-clean validation-best metric for a new holdout."""
    def __init__(self, *args, initial_stage=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_stage = bool(initial_stage)

    def load_state_dict(self, state_dict):
        if self.initial_stage:
            print('RESET_OLD_VALIDATION_CHECKPOINT_RANKINGS: new validation population',flush=True)
            return
        return super().load_state_dict(state_dict)


class StopAtSegmentEpochBoundary(Callback):
    """Segment-local timer, no checkpointed elapsed-time state.

    Stops *after* an epoch/validation, not halfway through an epoch.
    Reserve 4h before the NMRbox 48h allocation ceiling.
    """
    def __init__(self, hours):
        self.hours = float(hours)
        self.started = None

    def on_fit_start(self, trainer, pl_module):
        self.started = time.monotonic()

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or self.started is None:
            return
        elapsed = time.monotonic() - self.started
        if elapsed >= self.hours * 3600:
            if trainer.is_global_zero:
                print(f'SEGMENT_TIME_LIMIT: elapsed_hours={elapsed/3600:.3f}', flush=True)
            trainer.should_stop = True

    def on_train_epoch_end(self, trainer, pl_module):
        if self.started is not None and (time.monotonic() - self.started) >= self.hours * 3600:
            trainer.should_stop = True


class FinalizeSegment(Callback):
    """Save a full resumable checkpoint before DDP strategy teardown."""
    def __init__(self, folder, metadata, segment, batch, devices, code_sha, ckpt, best, timer):
        self.folder = folder
        self.metadata = metadata
        self.segment = segment
        self.batch = batch
        self.devices = devices
        self.code_sha = code_sha
        self.ckpt = ckpt
        self.best = best
        self.timer = timer

    def on_fit_end(self, trainer, pl_module):
        temp = self.folder / 'last.ckpt.tmp'
        trainer.save_checkpoint(str(temp), weights_only=False)
        trainer.strategy.barrier('segment_ckpt_written')
        if trainer.is_global_zero:
            if not temp.is_file() or temp.stat().st_size < 100 * 1024**2:
                raise RuntimeError(f'Final checkpoint missing/truncated: {temp}')
            final = self.folder / 'last.ckpt'
            os.replace(temp, final)
            summary = dict(
                checkpoint=str(final), checkpoint_bytes=final.stat().st_size,
                checkpoint_sha256=sha256_file(final),
                segment=self.segment, batch_size=self.batch, gpus=self.devices,
                epoch=int(trainer.current_epoch), global_step=int(trainer.global_step),
                target_epochs=int(self.metadata['target_epochs']),
                target_reached=(int(trainer.current_epoch) >= int(self.metadata['target_epochs'])),
                elapsed_hours=round((time.monotonic()-self.timer.started)/3600, 4),
                git_sha=self.code_sha, starting_checkpoint=str(self.ckpt),
                finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                best_model_path=str(self.best.best_model_path),
                best_model_score=(float(self.best.best_model_score)
                                  if self.best.best_model_score is not None else None),
            )
            pending = self.folder / 'result.json.tmp'
            pending.write_text(json.dumps(summary, indent=2)+'\n')
            os.replace(pending, self.folder/'result.json')
            print('SEGMENT_PASS', json.dumps(summary), flush=True)
        trainer.strategy.barrier('segment_done')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--campaign', required=True)
    p.add_argument('--segment', type=int, required=True)
    p.add_argument('--expected-gpus', type=int, choices=(1, 2), required=True)
    p.add_argument('--batch', type=int, choices=(32, 64, 128), required=True)
    p.add_argument('--segment-hours', type=float, default=44)
    args = p.parse_args()

    base = Path(args.base).resolve()
    repo = base / 'chefnmr'
    campaign = base / 'logs' / args.campaign
    folder = campaign / f'segment_{args.segment:02d}'
    folder.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((campaign/'campaign.json').read_text())
    if metadata['campaign'] != args.campaign or args.segment > metadata['segments']:
        raise RuntimeError('Segment/campaign mismatch')
    if sha256_file(Path(__file__)) != metadata['train_script_sha256']:
        raise RuntimeError('Pinned training launch script was modified after campaign creation')
    assert args.expected_gpus in (1, 2)
    devices = torch.cuda.device_count()
    if devices != args.expected_gpus:
        raise RuntimeError(f'Expected {args.expected_gpus} allocated GPUs, torch sees {devices}')
    print('GPU VISIBLE:', [torch.cuda.get_device_name(i) for i in range(devices)], flush=True)
    code_sha = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if code_sha != metadata['chef_commit']:
        raise RuntimeError(f'ChefNMR Git SHA drift: {code_sha} != {metadata["chef_commit"]}')
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise RuntimeError('ChefNMR tracked files dirty; refuse training')

    if args.segment == 1:
        ckpt = Path(metadata['original_checkpoint'])
    else:
        prev = campaign / f'segment_{args.segment-1:02d}' / 'last.ckpt'
        ckpt = prev
    # Preserve meaningful work if an OOM/requeue occurred after an hourly save.
    hourly = folder / 'hourly' / 'last.ckpt'
    if hourly.is_file() and hourly.stat().st_size > 100 * 1024**2:
        ckpt = hourly
    if not ckpt.is_file() or ckpt.stat().st_size < 100 * 1024**2:
        raise RuntimeError(f'Full resumable checkpoint missing: {ckpt}')
    if args.segment == 1 and ckpt.resolve() == Path(metadata['original_checkpoint']).resolve():
        expected = metadata.get('original_checkpoint_sha256')
        if expected and sha256_file(ckpt) != expected:
            raise RuntimeError('Pinned frozen Stage 2 continuation checkpoint SHA256 mismatch')
    print('RESUMING_FULL_LIGHTNING_CHECKPOINT:', ckpt, flush=True)

    cfg_path = repo / 'experiments/spec2mol/initial_training_2026_10_28/configs/augmented/config.yaml'
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_struct(cfg, False)
    cfg.dataset_args.datadir = os.environ.get('CHEFNMR_STAGE_DATADIR', str(base / 'data'))
    print('TRAINING_DATA_DIR:', cfg.dataset_args.datadir, flush=True)
    cfg.dataset_args.selection_manifest_dir = str(base/'selections/benchmark_molecule_disjoint_v1')
    cfg.dataset_args.spectral_variant_mode = 'paired_random'
    cfg.dataset_args.spectral_variant_count = 11
    cfg.dataset_args.spectral_variant_seed = 42
    cfg.dataset_args.batch_size = args.batch
    cfg.dataset_args.num_workers = 2
    cfg.dataset_args.pin_memory = True
    cfg.trainer.devices = devices
    cfg.trainer.num_nodes = 1
    cfg.trainer.max_epochs = int(metadata['target_epochs'])
    cfg.trainer.max_steps = -1
    cfg.trainer.check_val_every_n_epoch = 1
    cfg.trainer.num_sanity_val_steps = 2
    cfg.trainer.log_every_n_steps = 50
    cfg.general.mode = 'train'
    cfg.general.experiment_name = 'spec2mol-augmented-disjoint-stage2-v1'
    cfg.general.wandb = 'disabled'
    # Sampling/graph reconstruction during training is expensive, and final evaluation is separate.
    cfg.validation_args.sample_every_n_val_epoch = 1000000
    cfg.general.ckpt_abs_path = None  # The full state comes from Trainer.fit(ckpt_path=...)

    seed_everything(int(cfg.general.seed), workers=True)
    dm = NMRDataModule(cfg.dataset_args)
    dm.prepare_data()
    selection_path = base / 'selections/benchmark_molecule_disjoint_v1'
    selection_manifest = json.loads((selection_path / 'manifest.json').read_text())
    if (len(dm.train_indices) != selection_manifest['train_molecules']
        or len(dm.val_indices) != selection_manifest['validation_molecules']
        or len(dm.test_indices) != 0):
        raise RuntimeError('New disjoint split count differs from frozen manifest')
    if any(selection_manifest['audit'].get(k) != 0 for k in (
        'benchmark_exact_structure_overlap_train',
        'benchmark_exact_structure_overlap_val',
        'train_val_structure_identity_overlap',
        'previously_trained_validation_overlap')):
        raise RuntimeError('Benchmark leakage audit not PASS')
    if sha256_file(selection_path / 'manifest.json') != metadata['split_manifest_sha256']:
        raise RuntimeError('Training split manifest changed since submission')
    for part in ('train', 'val', 'test'):
        if sha256_file(selection_path / f'{part}.npy') != selection_manifest['split_file_sha256'][part]:
            raise RuntimeError(f'Training split {part}.npy changed since audit')
    if cfg.diffusion_process_args.edm_args.sigma_data is None:
        cfg.diffusion_process_args.edm_args.sigma_data = dm.sigma_data
    cfg.dataset_args.max_n_atoms = dm.max_n_atoms

    attempt_dir = folder / f'attempt_batch{args.batch}'
    attempt_dir.mkdir(exist_ok=True)
    current_rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))
    if current_rank == 0:
        OmegaConf.save(cfg, attempt_dir/'resolved_config.yaml')
    runtime = dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   git_sha=code_sha, host=socket.gethostname(), pid=os.getpid(),
                   segment=args.segment, batch=args.batch, n_gpus=devices,
                   model='ChefNMR-S', dataset='benchmark_molecule_disjoint_v1',
                   checkpoint=str(ckpt), ckpt_size=ckpt.stat().st_size)
    if current_rank == 0:
        (attempt_dir/'run.json').write_text(json.dumps(runtime, indent=2) + '\n')

    model = NMRTo3DStructureElucidation(cfg=cfg)
    # New validation population: checkpoint best score must not be carried over from
    # the original strict-clean experiment. Trainer.fit restores optimizer state.
    print('NEW_VALIDATION_POPULATION: previous val scores are not comparable',flush=True)
    # Keep best validation checkpoints throughout the campaign in one directory.
    best = FreshPopulationCheckpoint(initial_stage=(ckpt.resolve() == Path(metadata['original_checkpoint']).resolve()),
        dirpath=str(campaign/'best'),
        filename='epoch{epoch:05d}-step{step:09d}-loss{val/diffusion_loss:.2f}',
        monitor='val/diffusion_loss', auto_insert_metric_name=False,
        save_top_k=2, mode='min', every_n_epochs=1, save_last=False)
    # Independent recovery checkpoint about once per hour (also during a long epoch).
    hourly_callback = ModelCheckpoint(dirpath=str(folder/'hourly'),
        save_top_k=0, save_last=True,
        train_time_interval=timedelta(minutes=60))
    timer = StopAtSegmentEpochBoundary(min(args.segment_hours, float(metadata['segment_hours'])))
    trainer_config = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_config.pop('max_time', None)
    csv_logger = CSVLogger(save_dir=str(attempt_dir), name='metrics', version='v1')
    finalizer = FinalizeSegment(folder, metadata, args.segment, args.batch,
                                devices, code_sha, ckpt, best, timer)
    trainer = Trainer(
        strategy=(DDPStrategy(find_unused_parameters=bool(cfg.dataset_args.find_unused_parameters))
                  if devices > 1 else "auto"),
        callbacks=[best, hourly_callback, timer, finalizer],
        logger=csv_logger, enable_progress_bar=False,
        default_root_dir=str(attempt_dir), **trainer_config,
    )
    trainer.fit(model, datamodule=dm, ckpt_path=str(ckpt))
    if trainer.is_global_zero and not (folder / 'result.json').is_file():
        raise RuntimeError('Fit returned without finalized segment result')


if __name__ == '__main__':
    main()
