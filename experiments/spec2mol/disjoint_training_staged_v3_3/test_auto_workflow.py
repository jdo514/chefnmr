#!/usr/bin/env python3
"""CPU-only tests: no HTCondor submissions, GPUs, or production dataset access."""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import control
import patch_batch_policy


def write_result(root, segment, gpus, epoch, nbytes=101 * 1024 * 1024):
    folder = root / f'segment_{segment:02d}'
    folder.mkdir(exist_ok=True)
    checkpoint = folder/'last.ckpt'
    with checkpoint.open('wb') as f:
        f.seek(nbytes-1); f.write(b'X')
    meta = dict(segment=segment,gpus=gpus,epoch=epoch,
                checkpoint=str(checkpoint),checkpoint_bytes=nbytes,
                checkpoint_sha256=control.sha(checkpoint))
    (folder/'result.json').write_text(json.dumps(meta)+'\n')


def test_controller():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d)
        (root/'campaign.json').write_text(json.dumps({'segments':4,'target_epochs':100,
            'campaign_key':'a'*32})+'\n')
        assert control.pre(root,1)==0
        folder=root/'segment_01';folder.mkdir()
        claim=folder/'.winner_claim';claim.mkdir()
        (claim/'owner.json').write_text('{"gpus":2,"pid":12}')
        # Loser may finish before the winner imports torch or creates started.json.
        assert control.post(root,1,1,0,'111.0')==0
        # A failed winner must stop the DAG (not silently continue).
        assert control.post(root,1,2,1,'112.0')==20
        assert (folder/'.winner_failed').is_file()
        (folder/'.winner_failed').unlink() # synthetic fresh scenario
        write_result(root,1,2,11)
        with mock.patch.object(control,'cancel_sibling') as cancel:
            assert control.post(root,1,2,0,'112.0')==0
            cancel.assert_called_once()
        assert control.pre(root,2)==0
        # Any later segment should skip if target reached, even when an
        # intervening PRE skipped earlier segments and wrote no result.json.
        write_result(root,2,1,100)
        assert control.pre(root,3)==control.SKIP
        assert control.pre(root,4)==control.SKIP
        assert control.post(root,1,1,-1,'111.0')==0
        # Cancellation lookup must be narrowed to campaign+segment+other GPUs.
        meta={'campaign_key':'b'*32}
        with mock.patch.object(control.subprocess,'run',side_effect=[
            subprocess.CompletedProcess(['condor_q'],0,stdout='443 0\n',stderr=''),
            subprocess.CompletedProcess(['condor_rm'],0,stdout='',stderr='')]) as run:
            control.cancel_sibling(meta,1,2)
            expression=run.call_args_list[0].args[0][2]
            assert 'Spec2MolCampaignKey' in expression and 'Spec2MolSegment == 1' in expression
            assert 'Spec2MolCandidateGPUs == 1' in expression
            assert run.call_args_list[1].args[0]==['condor_rm','443.0']
        # Refuse a corrupt last.ckpt despite existing result.json.
        (folder/'last.ckpt').write_bytes(b'truncated')
        try: control.pre(root,2)
        except RuntimeError as e: assert 'checkpoint' in str(e).lower()
        else: raise AssertionError('Missing corrupt checkpoint detection')


def test_batch_patch():
    txt=(HERE/'run_disjoint.sh').read_text()
    assert 'BATCHES=(128 64 32)' in txt and 'BATCHES=(64 32)' in txt
    assert '"$FOLDER/.winner_claim"' in txt and '"$FOLDER/.winner.lock"' in txt
    assert '"$ROOT/.winner_claim"' not in txt
    assert patch_batch_policy.transform(txt)==txt
    old=txt.replace(patch_batch_policy.NEW_BATCH,patch_batch_policy.OLD_BATCH)
    old=old.replace('exec 9>"$FOLDER/.winner.lock"','exec 9>"$ROOT/.winner.lock"')
    old=old.replace('mkdir "$FOLDER/.winner_claim"','mkdir "$ROOT/.winner_claim"')
    assert patch_batch_policy.transform(old)==txt


def test_generator():
    with tempfile.TemporaryDirectory() as d:
        base=Path(d)/'spec2mol-chefnmr'
        (base/'chefnmr').mkdir(parents=True)
        (base/'launch/disjoint_training_staged_v3_3').mkdir(parents=True)
        for p in HERE.glob('*.py'):
            (base/'launch/disjoint_training_staged_v3_3'/p.name).write_bytes(p.read_bytes())
        (base/'launch/disjoint_training_staged_v3_3/run_disjoint.sh').write_bytes((HERE/'run_disjoint.sh').read_bytes())
        os.makedirs(base/'selections/benchmark_molecule_disjoint_v1',exist_ok=True)
        os.makedirs(base/'selections/strict_clean_conformers_v1',exist_ok=True)
        sel=base/'selections/benchmark_molecule_disjoint_v1'
        old=base/'selections/strict_clean_conformers_v1/train.npy'
        old.write_bytes(b'synthetic_old_train')
        for name in ('train','val','test'):
            (sel/f'{name}.npy').write_bytes(f'synthetic_{name}'.encode())
        mani=dict(audit={'benchmark_exact_structure_overlap_train':0,
                         'benchmark_exact_structure_overlap_val':0,
                         'train_val_structure_identity_overlap':0,
                         'previously_trained_validation_overlap':0},
                  benchmarks={'BMRB100':{'rows':100},'SpecTeach':{'rows':238}},
                  train_molecules=127481,validation_molecules=14165,
                  previous_training_ids_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
                  split_file_sha256={n:hashlib.sha256((sel/f'{n}.npy').read_bytes()).hexdigest()
                                     for n in ('train','val','test')})
        (sel/'manifest.json').write_text(json.dumps(mani))
        ckpt=base/'logs/full_augmented_2222972.0/checkpoints/spec2mol-augmented-seed42/diffusion_loss/last.ckpt'
        ckpt.parent.mkdir(parents=True)
        with ckpt.open('wb') as f:
            f.seek(101*1024*1024-1);f.write(b'A')
        subprocess.run(['git','init','-q',str(base/'chefnmr')],check=True)
        subprocess.run(['git','-C',str(base/'chefnmr'),'config','user.name','Test'],check=True)
        subprocess.run(['git','-C',str(base/'chefnmr'),'config','user.email','test@example.invalid'],check=True)
        (base/'chefnmr/README').write_text('synthetic test repo')
        subprocess.run(['git','-C',str(base/'chefnmr'),'add','.'],check=True)
        subprocess.run(['git','-C',str(base/'chefnmr'),'commit','-qm','test'],check=True)
        p=subprocess.run([sys.executable,str(HERE/'make_campaign.py'),'--base',str(base),
                          '--segments','3','--hours','1'],text=True,capture_output=True)
        assert p.returncode==0,p.stdout+p.stderr
        out=base/'logs/extended_augmented_disjoint_auto_v2'
        dag=(out/'campaign.dag').read_text()
        assert dag.count('JOB S0')==6
        assert 'PARENT S01_1 S01_2 CHILD S02_1 S02_2' in dag
        assert 'PARENT S02_1 S02_2 CHILD S03_1 S03_2' in dag
        assert dag.count('PRE_SKIP')==6 and dag.count('SCRIPT DEFER 75 60 POST')==6
        assert '+Spec2MolCampaignKey' in (out/'condor/S01_1.sub').read_text()
        meta=json.loads((out/'campaign.json').read_text())
        assert meta['model_selection_checkpoint_policy']=='best_top2_weights_only_every10_epochs'
        assert meta['recovery_checkpoint_policy']=='full_optimizer_hourly_60min_plus_final'
        assert 'disjoint_training_staged_v3_3' in (out/'condor/S01_1.sub').read_text()
        assert 'request_gpus = 2' in (out/'condor/S02_2.sub').read_text()
        assert 'request_gpus = 1' in (out/'condor/S02_1.sub').read_text()



def test_gpu_priority_patch():
    launcher=(HERE/'run_disjoint.sh').read_text()
    assert launcher.count('ONE_GPU_PRIORITY_WINDOW')==1
    assert 'if [[ "$EXPECTED_GPUS" -eq 1 ]]; then' in launcher
    assert launcher.index('sleep 60') < launcher.index('exec 9>"$FOLDER/.winner.lock"')
    assert launcher.index('sleep 60') < launcher.index('mkdir "$FOLDER/.winner_claim"')

if __name__=='__main__':
    test_gpu_priority_patch()
    test_batch_patch()
    test_controller()
    test_generator()
    print('AUTO-STAGE2 CPU TEST PASS: 60-second 2-GPU claim preference,  global batch patch, per-segment lock, '
          'one/two-GPU nodes, winner/loser handling, checkpoint audit, '
          'target PRE_SKIP, 3-segment DAG')
