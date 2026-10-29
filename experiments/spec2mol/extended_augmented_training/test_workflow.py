#!/usr/bin/env python3
"""CPU-only DAG/fallback tests; never starts HTCondor and never loads GPU data."""
import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('workflow', HERE/'workflow.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def run():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        (base/'chefnmr/experiments/spec2mol/initial_training_2026_10_28/configs/augmented').mkdir(parents=True)
        (base/'chefnmr/experiments/spec2mol/initial_training_2026_10_28/configs/augmented/config.yaml').write_text('test: true\n')
        initial = base/'logs/full_augmented_2222972.0/checkpoints/spec2mol-augmented-seed42/diffusion_loss/last.ckpt'
        initial.parent.mkdir(parents=True)
        with initial.open('wb') as f:
            f.truncate(110*1024*1024)
        (base/'data').mkdir()
        (base/'data/molecules.h5').touch()
        (base/'selections/strict_clean_conformers_v1').mkdir(parents=True)
        (base/'selections/strict_clean_conformers_v1/manifest.json').write_text('{}')
        (base/'venv/bin').mkdir(parents=True)
        (base/'venv/bin/python').touch()
        (base/'chefnmr/.git').mkdir()
        (base/'pipeline/.git').mkdir(parents=True)
        import subprocess
        oldbase = w.project_base
        oldcheck = subprocess.check_output
        w.project_base = lambda: base
        subprocess.check_output = lambda *args, **kwargs: 'test_commit\n'
        try:
            w.make(SimpleNamespace(campaign='test_001',segments=3,fallback_hours=6,
                                    segment_hours=44,target_epochs=10000))
        finally:
            subprocess.check_output = oldcheck
        root = base/'logs/test_001'
        dag = (root/'campaign.dag').read_text()
        assert 'PRE_SKIP S01_4 42' in dag
        assert 'PRE_SKIP S01_2 42' in dag
        assert 'PARENT S01_4 CHILD S01_2' in dag
        assert 'PARENT S01_2 CHILD S02_4' in dag
        assert 'periodic_remove = (JobStatus == 1)' in (root/'condor/S01_4.sub').read_text()
        assert 'periodic_remove' not in (root/'condor/S01_2.sub').read_text()
        a = SimpleNamespace(campaign='test_001',segment=1,return_code='-1')
        assert w.pre4(a) == 0
        assert w.post4(a) == 0  # never started; allow fallback
        assert w.pre2(a) == 0
        folder = w.seg_dir(root, 1)
        ckpt = folder/'last.ckpt'
        with ckpt.open('wb') as f: f.truncate(110*1024*1024)
        (folder/'result.json').write_text(json.dumps(dict(checkpoint=str(ckpt),gpus=2,
            batch_size=128,epoch=300,global_step=999,target_reached=False)))
        a.return_code='0'
        assert w.post2(a) == 0
        assert w.pre4(SimpleNamespace(campaign='test_001',segment=2)) == 0
        # A started and failing four-GPU job must never auto-fallback.
        a2 = SimpleNamespace(campaign='test_001',segment=2,return_code='1')
        folder2 = w.seg_dir(root, 2)
        folder2.mkdir()
        (folder2/'started.json').write_text('{}')
        try:
            w.post4(a2)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Must block fallback after a started failed job')
        (folder2/'started.json').unlink()
        assert w.post4(a2) == 0
        assert w.pre2(a2) == 0
        ckpt2 = folder2/'last.ckpt'
        with ckpt2.open('wb') as f: f.truncate(110*1024*1024)
        (folder2/'result.json').write_text(json.dumps(dict(checkpoint=str(ckpt2),gpus=2,
            batch_size=64,epoch=10000,global_step=15000,target_reached=True)))
        assert w.pre4(SimpleNamespace(campaign='test_001',segment=3)) == 42
        assert w.pre2(SimpleNamespace(campaign='test_001',segment=3)) == 42
        w.project_base=oldbase
    print('CPU WORKFLOW TESTS PASS: 4→2 fallback, no duplicate training, sequential checkpoint gating, target skip')

if __name__=='__main__': run()
