#!/usr/bin/env python3
"""CPU-only Lightning integration smoke for v3.4 timed-only recovery.

Runs small real Trainer fits (no GPU/data uploads). The assertions catch the
v3.3 save_last=True behavior which saves at epoch/validation boundaries.
"""
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import torch
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint


class TinyModel(LightningModule):
    def __init__(self, sleep_seconds=0.0):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.sleep_seconds = sleep_seconds

    def training_step(self, batch, batch_idx):
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        x, y = batch
        loss = torch.nn.functional.mse_loss(self.linear(x), y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.log('val_loss', torch.nn.functional.mse_loss(self.linear(x), y))

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
        return {'optimizer': opt, 'lr_scheduler': scheduler}


class CountTimedSaves(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_calls = []

    def _save_checkpoint(self, trainer, filepath):
        super()._save_checkpoint(trainer, filepath)
        self.save_calls.append((trainer.current_epoch, trainer.global_step, filepath))


def recovery_callback(folder, interval):
    return CountTimedSaves(
        dirpath=str(folder), filename='last', save_top_k=1,
        save_last=False, train_time_interval=interval,
        every_n_epochs=0, save_on_train_epoch_end=False,
        save_weights_only=False, enable_version_counter=False,
    )


def do_fit(callback, epochs, batches, pause):
    xs = torch.arange(batches, dtype=torch.float32).view(-1, 1) / max(1, batches)
    ds = TensorDataset(xs, xs.clone())
    loader = DataLoader(ds, batch_size=1, num_workers=0)
    model = TinyModel(sleep_seconds=pause)
    trainer = Trainer(
        accelerator='cpu', devices=1, max_epochs=epochs, callbacks=[callback],
        logger=False, enable_progress_bar=False, enable_model_summary=False,
        enable_checkpointing=True, limit_val_batches=1, num_sanity_val_steps=0,
        log_every_n_steps=100,
    )
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)


def main():
    import lightning
    print('Lightning runtime:', lightning.__version__)
    with TemporaryDirectory(prefix='chefnmr_v34_timed_ckpt_') as td:
        root = Path(td)
        # Fast epochs and validation must NEVER produce checkpoints when the
        # time interval has not passed.
        cb = recovery_callback(root/'no_save', timedelta(minutes=60))
        do_fit(cb, epochs=3, batches=3, pause=0.0)
        assert cb.save_calls == [], f'Unexpected epoch/validation saves: {cb.save_calls}'
        assert not list((root/'no_save').glob('*.ckpt')), 'Early checkpoint appeared'
        print('TIMED_ONLY_NO_EPOCH_SAVES_PASS')

        # A short timer must trigger a genuine full resume file on train batch end.
        cb = recovery_callback(root/'timed', timedelta(milliseconds=150))
        do_fit(cb, epochs=2, batches=8, pause=0.08)
        assert cb.save_calls, 'Timed checkpoint never fired'
        checkpoints = list((root/'timed').glob('*.ckpt'))
        assert len(checkpoints)==1 and checkpoints[0].name=='last.ckpt', checkpoints
        assert all(Path(p).name=='last.ckpt' for _,_,p in cb.save_calls)
        checkpoint = torch.load(checkpoints[0], map_location='cpu', weights_only=False)
        assert checkpoint.get('optimizer_states'), 'Full optimizer state absent'
        assert checkpoint.get('lr_schedulers'), 'Scheduler state absent'
        assert checkpoint.get('loops'), 'Lightning loop state absent'
        print('TIMED_ONLY_FULL_RESUME_CHECKPOINT_PASS:',len(cb.save_calls),'saves;',
              checkpoints[0].stat().st_size,'bytes')

if __name__ == '__main__':
    main()
