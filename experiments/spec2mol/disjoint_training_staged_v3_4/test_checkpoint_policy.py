#!/usr/bin/env python3
"""Static audit: best weights-only every 10; full-state timed-only hourly."""
import ast
from pathlib import Path

src=(Path(__file__).parent/'train_disjoint.py').read_text()
tree=ast.parse(src)
calls={n.func.id: n for n in ast.walk(tree)
       if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
       and n.func.id in ('FreshPopulationCheckpoint','ModelCheckpoint')}
assert set(calls)=={'FreshPopulationCheckpoint','ModelCheckpoint'}
kw=lambda node: {k.arg:k.value for k in node.keywords}
b=kw(calls['FreshPopulationCheckpoint'])
h=kw(calls['ModelCheckpoint'])
value=lambda x:ast.literal_eval(x)
assert value(b['save_weights_only']) is True
assert value(b['every_n_epochs'])==10
assert value(b['save_top_k'])==2
assert value(b['monitor'])=='val/diffusion_loss'
assert value(h['filename'])=='last'
assert value(h['save_weights_only']) is False
assert value(h['save_last']) is False
assert value(h['save_top_k'])==1
assert value(h['every_n_epochs'])==0
assert value(h['save_on_train_epoch_end']) is False
assert value(h['enable_version_counter']) is False
assert h['train_time_interval'].func.id=='timedelta'
assert value(kw(h['train_time_interval'])['minutes'])==60
assert 'trainer.save_checkpoint(str(temp), weights_only=False)' in src
assert 'trainer.fit(model, datamodule=dm, ckpt_path=str(ckpt))' in src
assert "if sha256_file(Path(__file__)) != metadata['train_script_sha256']" in src
print('CHECKPOINT_POLICY_V34_TEST_PASS: no epoch-based full recovery writes; '
      'time-trigger only, path hourly/last.ckpt, full resume state')
