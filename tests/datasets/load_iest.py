import unittest

from tklearn.datasets import load_iest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        x, y, z = load_iest('D:\Documents\Resources\Datasets\IEST-2018')
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
