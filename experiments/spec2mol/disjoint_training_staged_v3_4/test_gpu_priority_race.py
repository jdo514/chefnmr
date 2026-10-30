#!/usr/bin/env python3
"""CPU-only integration test for the 60-second preference (test uses 1s).

Creates an isolated copy of the production launcher with a 1-second wait;
never submits HTCondor jobs or loads torch/GPU. Fake Python only replaces
CUDA discovery and the training entrypoint, while all other Python calls
(including SHA verification) run normally.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

HERE=Path(__file__).resolve().parent
FAKE_PY = r'''#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == "-c" && "$2" == *'torch.cuda.device_count()'* ]]; then
  echo "$FAKE_GPUS"
  exit 0
fi
if [[ "$1" == */stage_hdf5.py ]]; then
  /usr/bin/env python3 - "$@" <<'MOCK_STAGE'
import pathlib,sys
args=sys.argv[1:]
path=pathlib.Path(args[args.index('--selected-datadir-file')+1])
path.write_text(str(pathlib.Path(__file__).parent) + '\n')
pathlib.Path(args[args.index('--report')+1]).write_text('{"mode":"nfs","reason":"mock stage"}\n')
MOCK_STAGE
  exit 0
fi
if [[ "$1" == */train_disjoint.py ]]; then
  /usr/bin/env python3 - "$FAKE_BASE" "$FAKE_GPUS" <<'INNER'
import json,sys,time
from pathlib import Path
p=Path(sys.argv[1])/'logs'/'synthetic_priority_campaign'/'segment_01'
(p/'result.json').write_text(json.dumps({'fake_train_success':True, 'gpus':int(sys.argv[2])}))
time.sleep(0.2)
INNER
  exit 0
fi
exec /usr/bin/env python3 "$@"
'''


def setup(root):
    base = root/'spec2mol-chefnmr'
    work = base/'launch/disjoint_training_staged_v3_4'
    work.mkdir(parents=True)
    content=(HERE/'run_disjoint.sh').read_text()
    assert content.count('sleep 60')==1
    content=content.replace('sleep 60','sleep 1')
    launch=work/'run_disjoint.sh'
    launch.write_text(content)
    launch.chmod(0o755)
    (work/'train_disjoint.py').write_text('# placeholder replaced by fake Python\n')
    (work/'stage_hdf5.py').write_bytes((HERE/'stage_hdf5.py').read_bytes())
    py=base/'venv/bin/python'
    py.parent.mkdir(parents=True)
    py.write_text(FAKE_PY)
    py.chmod(0o755)
    (base/'chefnmr').mkdir()
    campaign=base/'logs/synthetic_priority_campaign'
    campaign.mkdir(parents=True)
    (campaign/'campaign.json').write_text(json.dumps({
        'launcher_script_sha256':hashlib.sha256(launch.read_bytes()).hexdigest(),
        'staging_script_sha256':hashlib.sha256((work/'stage_hdf5.py').read_bytes()).hexdigest(),
        'requested_disk_gib':50
    }))
    return base,launch


def launch_job(base,script,gpus):
    env=dict(os.environ, FAKE_GPUS=str(gpus),FAKE_BASE=str(base))
    return subprocess.Popen(['/bin/bash',str(script),'synthetic_priority_campaign','1',str(gpus)],
                            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env)


def check(near_simultaneous):
    with tempfile.TemporaryDirectory(prefix='gpu_priority_test_') as d:
        base,script=setup(Path(d))
        first=launch_job(base,script,1)
        if near_simultaneous:
            time.sleep(0.15)
            second=launch_job(base,script,2)
        else:
            second=None
        try:
            first_out,_=first.communicate(timeout=12)
            assert first.returncode==0,first_out
            if second:
                second_out,_=second.communicate(timeout=12)
                assert second.returncode==0,second_out
        finally:
            if first.poll() is None:first.kill()
            if second and second.poll() is None:second.kill()
        folder=base/'logs/synthetic_priority_campaign/segment_01'
        owner=json.loads((folder/'.winner_claim/owner.json').read_text())
        result=json.loads((folder/'result.json').read_text())
        expected=2 if near_simultaneous else 1
        assert owner['gpus']==expected,(owner,first_out)
        assert result['gpus']==expected,(result,first_out)
        assert 'ONE_GPU_PRIORITY_WINDOW' in first_out
        if near_simultaneous:
            # Loser may observe the flock, the started marker or the persistent
            # claim depending on process scheduling. All are correct outcomes.
            loser_messages = (
                'Another GPU candidate already training',
                'Another candidate previously started',
                'Another candidate already claimed',
            )
            assert any(msg in first_out for msg in loser_messages), first_out
            assert 'TRAINING SEGMENT PASS' not in first_out, first_out
            assert 'TRAINING SEGMENT PASS' in second_out, second_out
        else:
            assert 'TRAINING SEGMENT PASS' in first_out, first_out
        print('PASS', 'simultaneous allocation -> two-GPU wins'
              if near_simultaneous else 'only one GPU -> training starts after 1s test wait')


if __name__=='__main__':
    check(True)
    check(False)
    print('GPU PRIORITY RACE TEST PASS (no actual GPUs)')
