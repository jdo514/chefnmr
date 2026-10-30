#!/usr/bin/env python3
"""CPU-only simulation: local verified copy, insufficient disk fallback, wrong hash fallback."""
import hashlib, json, tempfile
from pathlib import Path
import stage_hdf5 as mod

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    src = root / 'dataset' / 'molecules.h5'
    src.parent.mkdir()
    src.write_bytes(b'SPEC2MOL-TEST' * 20000)
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    scratch = root / 'scratch'
    scratch.mkdir()
    # min margins set to zero for small synthetic fixture
    a = mod.stage(src, scratch, request_disk_kib=3 * 1024 * 1024,
                  expected_sha=sha, min_spare_gib=0, min_alloc_spare_gib=0)
    assert a['mode'] == 'local', a
    assert Path(a['datadir'], 'molecules.h5').read_bytes() == src.read_bytes()
    assert not (scratch / 'spec2mol_chefnmr_h5_stage' / 'molecules.h5.incomplete').exists()
    (scratch / 'spec2mol_chefnmr_h5_stage' / 'molecules.h5').unlink()
    a = mod.stage(src, scratch, request_disk_kib=0,
                  expected_sha=sha, min_spare_gib=0, min_alloc_spare_gib=0)
    assert a['mode'] == 'nfs' and 'requested disk' in a['reason']
    a = mod.stage(src, scratch, request_disk_kib=3 * 1024 * 1024,
                  expected_sha='0'*64, min_spare_gib=0, min_alloc_spare_gib=0)
    assert a['mode'] == 'nfs' and 'SHA256' in a['reason'], a
    assert not (scratch / 'spec2mol_chefnmr_h5_stage' / 'molecules.h5').exists()
print('STAGE_HDF5 CPU TEST PASS: copy, checksum, disk guard, NFS fallback')
