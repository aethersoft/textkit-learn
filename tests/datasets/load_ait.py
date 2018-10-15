import unittest

from tklearn.datasets.load_ait import load_ait


class MyTestCase(unittest.TestCase):
    def test_E_c_ds(self):
        train, dev, test = load_ait('D:\Documents\Resources\\tkresources\datasets\SemEval-2018', task='E.c')
        self.assertTrue(True, True)

    def test_EI_oc_ds(self):
        train, dev, test = load_ait('D:\Documents\Resources\\tkresources\datasets\SemEval-2018', task='EI.oc')
        self.assertTrue(True, True)

    def test_EI_reg_ds(self):
        train, dev, test = load_ait('D:\Documents\Resources\\tkresources\datasets\SemEval-2018', task='EI.reg')
        self.assertTrue(True, True)

    def test_V_reg_ds(self):
        train, dev, test = load_ait('D:\Documents\Resources\\tkresources\datasets\SemEval-2018', task='V.reg')
        self.assertTrue(True, True)

    def test_V_oc_ds(self):
        train, dev, test = load_ait('D:\Documents\Resources\\tkresources\datasets\SemEval-2018', task='V.oc')
        self.assertTrue(True, True)


if __name__ == '__main__':
    unittest.main()
