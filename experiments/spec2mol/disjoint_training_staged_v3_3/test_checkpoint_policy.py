#!/usr/bin/env python3
"""Static, import-free verification of v3.3 checkpoint safety policy."""
import ast
from pathlib import Path
src=(Path(__file__).parent/'train_disjoint.py').read_text()
tree=ast.parse(src)
call=None
hourly=None
for n in ast.walk(tree):
    if isinstance(n,ast.Call) and isinstance(n.func,ast.Name):
        if n.func.id=='FreshPopulationCheckpoint':call=n
        if n.func.id=='ModelCheckpoint':hourly=n
assert call is not None and hourly is not None
kw=lambda node:{k.arg:k.value for k in node.keywords}
b=kw(call)
h=kw(hourly)
assert ast.literal_eval(b['save_weights_only']) is True
assert ast.literal_eval(b['every_n_epochs'])==10
assert ast.literal_eval(b['save_top_k'])==2
assert ast.literal_eval(b['monitor'])=='val/diffusion_loss'
assert ast.literal_eval(h['save_weights_only']) is False
assert ast.literal_eval(h['save_last']) is True
assert ast.literal_eval(h['save_top_k'])==0
assert isinstance(h['train_time_interval'],ast.Call)
assert isinstance(h['train_time_interval'].func,ast.Name)
assert h['train_time_interval'].func.id=='timedelta'
assert ast.literal_eval(kw(h['train_time_interval'])['minutes'])==60
assert 'trainer.save_checkpoint(str(temp), weights_only=False)' in src
assert 'trainer.fit(model, datamodule=dm, ckpt_path=str(ckpt))' in src
assert "if sha256_file(Path(__file__)) != metadata['train_script_sha256']" in src
print('CHECKPOINT_POLICY_V33_TEST_PASS: best weights-only every 10; hourly+final full; restore pinned full')
