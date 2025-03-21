import unittest
from numpy.testing import assert_almost_equal
from faultree import eval_fta, eval_rta
from test_data import fta1, rta1, fta2, rta2, fta3, rta3, fta4, rta4, \
    fta1_result, rta1_result, fta2_result, rta2_result, fta3_result, rta3_result, fta4_result, rta4_result

class TestFTA(unittest.TestCase):
    def test_fta1(self):
        test_result = eval_fta(fta1)
        self.assertAlmostEqual(test_result, fta1_result)

    def test_fta2(self):
        test_result = eval_fta(fta2)
        self.assertAlmostEqual(test_result, fta2_result)

    def test_fta3(self):
        test_result = eval_fta(fta3)
        assert_almost_equal(test_result, fta3_result)

    def test_fta4(self):
        test_result = eval_fta(fta4)
        assert_almost_equal(test_result, fta4_result)


class TestRTA(unittest.TestCase):
    def test_rta1(self):
        test_result = eval_rta(rta1)
        self.assertAlmostEqual(test_result, rta1_result)

    def test_rta2(self):
        test_result = eval_rta(rta2)
        self.assertAlmostEqual(test_result, rta2_result)

    def test_rta3(self):
        test_result = eval_rta(rta3)
        assert_almost_equal(test_result, rta3_result)

    def test_rta4(self):
        test_result = eval_rta(rta4)
        assert_almost_equal(test_result, rta4_result)


if __name__ == '__main__':
    unittest.main()
