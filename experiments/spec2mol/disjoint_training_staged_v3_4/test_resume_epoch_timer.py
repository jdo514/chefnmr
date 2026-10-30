#!/usr/bin/env python3
"""Stdlib-only smoke test of the launcher callback, no torch imported."""
import ast
import time
from pathlib import Path
src = (Path(__file__).parent / 'train_disjoint.py').read_text()
tree = ast.parse(src)
cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'ResumeSafeEpochTimer')
class Callback: pass
scope = {'Callback': Callback, 'time': time}
exec(compile(ast.Module(body=[cls],type_ignores=[]), '<timer>', 'exec'), scope)
class DummyTrainer: is_global_zero = True
class DummyModule:
    start_epoch_time = None
model = DummyModule()
scope['ResumeSafeEpochTimer']().on_fit_start(DummyTrainer(), model)
assert isinstance(model.start_epoch_time, float), model.start_epoch_time
prior = model.start_epoch_time
scope['ResumeSafeEpochTimer']().on_fit_start(DummyTrainer(), model)
assert model.start_epoch_time == prior
assert 'callbacks=[best, hourly_callback, ResumeSafeEpochTimer(), timer, finalizer]' in src
print('RESUME TIMER TEST PASS')
