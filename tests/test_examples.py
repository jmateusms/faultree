import unittest
import json
import os
import numpy as np
from faultree.builder import build, compute_event_probabilities

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), '..', 'examples')

class TestExamples(unittest.TestCase):

    def _load_and_run(self, filename, success_mode=False):
        path = os.path.join(EXAMPLES_DIR, filename)
        with open(path, 'r') as f:
            tree = json.load(f)
        
        bdd, top, expr, symb = build(tree, success_mode=success_mode)
        probs = compute_event_probabilities(bdd, tree, None, success_mode=success_mode)
        top_id = str(tree["id"]) if "id" in tree else None
        
        # Handle flat tree root finding logic implicitly handled by normalize_tree called inside build/compute
        # But we need the ID to look up the result.
        # normalize_tree is called inside compute_event_probabilities, but we don't get the normalized tree back.
        # We need to call normalize_tree manually to get the root ID if it's a flat tree.
        from faultree.builder import normalize_tree
        tree_norm = normalize_tree(tree)
        top_id = str(tree_norm["id"])
        
        return probs[top_id], tree_norm

    def test_basic_tree(self):
        # TE (OR) -> IE1 (AND: BE1, BE2), IE2 (AND: BE1, BE3)
        # BE1=0.12, BE2=0.07, BE3=0.09
        # Expected: 0.018444
        prob, _ = self._load_and_run("basic_tree.json")
        self.assertAlmostEqual(prob, 0.018444, places=6)

    def test_fta3(self):
        # A = OR(AND(1, 2), 3) (piecewise over samples)
        prob, _ = self._load_and_run("fta3.json")
        np.testing.assert_allclose(prob, np.array([0.02485, 0.069, 0.14975]), rtol=0, atol=1e-12)

    def test_ref_example(self):
        # TE (AND) -> IE (OR: BE1, BE2), BE1
        # Simplifies to BE1.
        # BE1=0.1
        prob, _ = self._load_and_run("ref_example.json")
        self.assertAlmostEqual(prob, 0.1, places=6)

    def test_connector(self):
        # Flattened OR of all BEs.
        # BEs: BE1(0.000601), BE2(0.000925), BE3(0.000139), BE4(0.000092), 
        #      BE5(0.000925), BE6(0.000092), BE8(0.000509)
        # Prob = 1 - product(1-P)
        probs = [0.000601, 0.000925, 0.000139, 0.000092, 0.000925, 0.000092, 0.000509]
        
        p_success = 1.0
        for p in probs:
            p_success *= (1.0 - p)
        expected = 1.0 - p_success
        
        prob, _ = self._load_and_run("connector.json")
        self.assertAlmostEqual(prob, expected, places=6)

    def test_flat_tree(self):
        # Just regression test this one or verify consistency.
        # Based on previous run, success mode was 0.996721
        # Fail mode should be 1 - 0.996721 = 0.003279 (approx)
        # Let's calculate exactly or use the value obtained.
        
        # BEs from file:
        # LPEI: 0.000601
        # CEIF: 0.000925
        # UCD:  0.000139
        # WJCD: 0.000092
        # LEC:  0.000925
        # FPEC: 0.000092
        # WDI:  0.000509
        
        # Structure:
        # LoIRC2tE_FT (OR)
        #  - LPEI
        #  - CEIF
        #  - G_CCLIR (OR: UCD, WJCD)
        #  - G_PE (OR: LEC, FPEC)
        #  - G_HCR (OR: UCD, WDI)
        
        # Note UCD is repeated.
        # TE = OR(LPEI, CEIF, UCD, WJCD, LEC, FPEC, UCD, WDI)
        #    = OR(LPEI, CEIF, UCD, WJCD, LEC, FPEC, WDI)
        # It is just an OR of all unique BEs.
        
        probs_map = {
             "LPEI": 0.000601,
             "CEIF": 0.000925,
             "UCD":  0.000139,
             "WJCD": 0.000092,
             "LEC":  0.000925,
             "FPEC": 0.000092,
             "WDI":  0.000509 
        }
        
        p_success = 1.0
        for p in probs_map.values():
            p_success *= (1.0 - p)
        expected = 1.0 - p_success
        
        prob, _ = self._load_and_run("flat_tree.json")
        self.assertAlmostEqual(prob, expected, places=6)

    def test_fta3_success(self):
        # Reliability Mode (Dual Tree): A = AND(OR(1, 2), 3) (piecewise over samples)
        prob, _ = self._load_and_run("fta3_success.json", success_mode=True)
        np.testing.assert_allclose(prob, np.array([0.97515, 0.931, 0.85025]), rtol=0, atol=1e-12)

    def test_flat_tree_success(self):
        # Reliability Mode.
        # Same structure as flat_tree (OR of all unique BEs), but transformed to AND of all unique BEs.
        # BEs provided are reliabilities.
        # TE = AND(LPEI, CEIF, UCD, WJCD, LEC, FPEC, WDI)
        # Prob = Product of all unique probabilities.
        
        probs_map = {
             "LPEI": 0.999399,
             "CEIF": 0.999075,
             "UCD":  0.999861,
             "WJCD": 0.999908,
             "LEC":  0.999075,
             "FPEC": 0.999908,
             "WDI":  0.999491 
        }
        
        expected = 1.0
        for p in probs_map.values():
            expected *= p
            
        prob, _ = self._load_and_run("flat_tree_success.json", success_mode=True)
        self.assertAlmostEqual(prob, expected, places=6)

if __name__ == '__main__':
    unittest.main()
