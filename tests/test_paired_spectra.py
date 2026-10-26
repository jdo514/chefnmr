"""Run: python -m unittest discover -s tests -p test_paired_spectra.py -v"""
import multiprocessing
import unittest
from types import SimpleNamespace
import numpy as np
from src.data.datamodule import NMRDataset
from src.data.paired_spectra import (
    resolve_item, select_paired_spectra, SpectralVariantEpochCallback,
)

class FakeFactory:
    def get_molecule_data_by_idx(self, mol_idx, **kwargs):
        n = 11
        return {'smiles': 'CC', 'spectra': {
            'h_10k': np.broadcast_to(np.arange(n, dtype=np.float32)[:, None], (n, 10000)).copy(),
            'c_80': np.broadcast_to((100 + np.arange(n, dtype=np.float32))[:, None], (n, 80)).copy(),
        }, 'atom_features': {}}

class FakeGenerator:
    def forward(self, data):
        return int(data['spectra']['h_10k'][0, 0]), int(data['spectra']['c_80'][0, 0])


def dataset(mode, count=11):
    ds = NMRDataset.__new__(NMRDataset)
    ds.data_indices = np.array([10, 11, 12], dtype=np.int32)
    ds.spectral_variant_mode = mode
    ds.spectral_variant_count = count
    ds.spectral_variant_seed = 42
    ds._spectral_epoch = multiprocessing.Value('i', 0)
    ds.data_manager = FakeFactory()
    ds.input_generator = FakeGenerator()
    return ds

class TestPairedSpectra(unittest.TestCase):
    def test_pair_and_reference(self):
        ds = dataset('reference')
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds[0], (0, 100))
        self.assertEqual(ds[2], (0, 100))

    def test_cyclic_11_epoch_coverage(self):
        ds = dataset('paired_cyclic')
        seen = set()
        for epoch in range(11):
            ds.set_spectral_epoch(epoch)
            h, c = ds[1]
            self.assertEqual(c, 100 + h)
            seen.add(h)
        self.assertEqual(seen, set(range(11)))

    def test_expanded_coverage(self):
        ds = dataset('paired_expanded')
        self.assertEqual(len(ds), 33)
        self.assertEqual([ds[i] for i in range(11)], [(i, 100 + i) for i in range(11)])
        self.assertEqual(ds[11], (0, 100))

    def test_mismatch_raises(self):
        mol = FakeFactory().get_molecule_data_by_idx(10)
        mol['spectra']['c_80'] = mol['spectra']['c_80'][:10]
        with self.assertRaises(ValueError):
            select_paired_spectra(mol, 1, 11)

    def test_worker_epoch_sharing(self):
        from torch.utils.data import DataLoader
        ds = dataset('paired_cyclic')
        loader = DataLoader(ds, batch_size=1, num_workers=2, persistent_workers=True)
        ds.set_spectral_epoch(0)
        out0 = [int(pair[0][0]) for pair in loader]
        ds.set_spectral_epoch(1)
        out1 = [int(pair[0][0]) for pair in loader]
        self.assertEqual(out1, [(v + 1) % 11 for v in out0])
        del loader

    def test_epoch_callback(self):
        ds = dataset('paired_cyclic')
        trainer = SimpleNamespace(current_epoch=17, datamodule=SimpleNamespace(set_spectral_epoch=ds.set_spectral_epoch))
        SpectralVariantEpochCallback().on_train_epoch_start(trainer, None)
        self.assertEqual(ds._spectral_epoch.value, 17)
        self.assertEqual(ds[0][0], (42 + 17) % 11)

    def test_no_legacy_pairing(self):
        self.assertEqual(resolve_item(1, 3, 11, 'legacy_random'), (1, None))

if __name__ == '__main__':
    unittest.main()
