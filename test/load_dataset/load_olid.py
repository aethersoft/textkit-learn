import unittest

from tklearn.datasets import load_olid, load_dataset


class LoadDatasetTestCase(unittest.TestCase):
    def test_load_olid_train_ds_subtask_a(self):
        ds = load_dataset('olid', split='train', task='subtask_a')
        self.assertEqual(len(ds.columns), 3)
        self.assertGreater(len(ds), 0)

    def test_load_olid_train_ds_subtask_b(self):
        ds = load_dataset('olid', split='train', task='subtask_b')
        self.assertEqual(len(ds.columns), 3)
        self.assertGreater(len(ds), 0)

    def test_load_olid_train_ds_subtask_c(self):
        ds = load_dataset('olid', split='train', task='subtask_c')
        self.assertEqual(len(ds.columns), 3)
        self.assertGreater(len(ds), 0)

    def test_load_olid_test_ds_subtask_a(self):
        ds = load_dataset('olid',  split='test', task='subtask_a')
        self.assertEqual(len(ds.columns), 3)
        self.assertGreater(len(ds), 0)

    def test_load_olid_test_ds_subtask_b(self):
        ds = load_dataset('olid', split='test', task='subtask_b')
        print(ds.head())
        self.assertEqual(len(ds.columns), 3)
        self.assertGreater(len(ds), 0)

    def test_load_olid_test_ds_subtask_c(self):
        ds = load_dataset('olid', split='test', task='subtask_c')
        self.assertEqual(len(ds.columns), 3)
        self.assertGreater(len(ds), 0)


if __name__ == '__main__':
    unittest.main()
