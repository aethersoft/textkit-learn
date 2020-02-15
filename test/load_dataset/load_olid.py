import unittest

from tklearn.datasets import load_olid, load_dataset


class LoadDatasetTestCase(unittest.TestCase):
    def test_olid_ds_load_subtask_a(self):
        ds = load_dataset('olid', task='subtask_a')
        self.assertEqual(len(ds.columns), 3)

    def test_olid_ds_load_subtask_b(self):
        ds = load_dataset('olid', task='subtask_b')
        self.assertEqual(len(ds.columns), 3)

    def test_olid_ds_load_subtask_c(self):
        ds = load_dataset('olid', task='subtask_c')
        self.assertEqual(len(ds.columns), 3)


if __name__ == '__main__':
    unittest.main()
