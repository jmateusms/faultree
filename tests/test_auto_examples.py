import unittest
import json
import os
import numpy as np
from faultree.builder import build, compute_event_probabilities, load_probs_from_file, normalize_tree, var_probs_from_tree, gather_nodes

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), '..', 'examples')

class TestExamples(unittest.TestCase):

    def _load_tree(self, filename):
        path = os.path.join(EXAMPLES_DIR, filename)
        with open(path, 'r') as f:
            return json.load(f)

    def _resolve_prob_input(self, filename, tree):
        if filename == "large_tree.json":
            return os.path.join(EXAMPLES_DIR, "large_example_probs.csv")
        if isinstance(tree, dict) and "prob_file" in tree and tree["prob_file"]:
            return os.path.join(EXAMPLES_DIR, str(tree["prob_file"]))
        return None

    def _run_example(self, filename, success_mode=False):
        tree = self._load_tree(filename)
        prob_input = self._resolve_prob_input(filename, tree)
        bdd, _, _, _ = build(tree, success_mode=success_mode)
        probs = compute_event_probabilities(bdd, tree, prob_input, success_mode=success_mode)
        tree_norm = normalize_tree(tree)
        top_id = str(tree_norm["id"])
        return probs[top_id], tree_norm, prob_input

    def _collect_basic_vars(self, node, out):
        if isinstance(node, dict) and "ref" in node:
            return
        children = node.get("children", []) if isinstance(node, dict) else []
        gate = node.get("gate") if isinstance(node, dict) else None
        if not children and gate in (None, "BASIC"):
            out.add(str(node["id"]))
            return
        for ch in children:
            self._collect_basic_vars(ch, out)

    def _eval_node(self, node, registry, assign, success_mode=False):
        if success_mode:
            # Definitional dual, sharing no gate-dualization logic with the
            # engine: success indicators y relate to failure indicators by
            # x = not y, and the system succeeds iff the failure tree does
            # not fire.
            fail_assign = {k: (not v) for k, v in assign.items()}
            return not self._eval_node_fail(node, registry, fail_assign)
        return self._eval_node_fail(node, registry, assign)

    def _eval_node_fail(self, node, registry, assign):
        if "ref" in node:
            target = registry[str(node["ref"])]
            return self._eval_node_fail(target, registry, assign)

        children = node.get("children", [])
        gate = node.get("gate")
        if not children and gate in (None, "BASIC"):
            return bool(assign[str(node["id"])])

        if gate == "AND":
            return all(self._eval_node_fail(ch, registry, assign) for ch in children)
        if gate == "OR":
            return any(self._eval_node_fail(ch, registry, assign) for ch in children)
        if gate == "XOR":
            trues = sum(self._eval_node_fail(ch, registry, assign) for ch in children)
            return trues == 1
        if gate == "K_OF_N":
            k = node.get("k")
            trues = sum(self._eval_node_fail(ch, registry, assign) for ch in children)
            return trues >= k

        raise ValueError(f"Unsupported gate: {gate}")

    def _bruteforce_probability(self, tree_norm, prob_input, success_mode=False):
        base_probs = var_probs_from_tree(tree_norm)
        if prob_input:
            file_probs = load_probs_from_file(prob_input)
            base_probs.update(file_probs)

        base_probs = {k: (np.array(v) if isinstance(v, list) else v) for k, v in base_probs.items()}

        vars_set = set()
        self._collect_basic_vars(tree_norm, vars_set)
        vars_list = sorted(vars_set)

        if any(isinstance(base_probs.get(v), np.ndarray) for v in vars_list):
            lengths = {len(base_probs[v]) for v in vars_list if isinstance(base_probs.get(v), np.ndarray)}
            if len(lengths) != 1:
                raise ValueError("Mismatched sample lengths in brute force evaluator")
            sample_len = next(iter(lengths))
            out = np.zeros(sample_len, dtype=float)
            for idx in range(sample_len):
                scalar_probs = {}
                for v in vars_list:
                    pv = base_probs.get(v, 0.0)
                    scalar_probs[v] = float(pv[idx]) if isinstance(pv, np.ndarray) else float(pv)
                out[idx] = self._bruteforce_probability_scalar(tree_norm, scalar_probs, vars_list, success_mode=success_mode)
            return out

        scalar_probs = {v: float(base_probs.get(v, 0.0)) for v in vars_list}
        return self._bruteforce_probability_scalar(tree_norm, scalar_probs, vars_list, success_mode=success_mode)

    def _bruteforce_probability_scalar(self, tree_norm, scalar_probs, vars_list, success_mode=False):
        if len(vars_list) > 18:
            raise ValueError("Too many basic events for brute force evaluator")

        registry = {}
        gather_nodes(tree_norm, registry)

        total = 0.0
        n = len(vars_list)
        for mask in range(1 << n):
            assign = {}
            p = 1.0
            for i, var in enumerate(vars_list):
                bit = bool(mask & (1 << i))
                assign[var] = bit
                pv = scalar_probs[var]
                p *= pv if bit else (1.0 - pv)
            if self._eval_node(tree_norm, registry, assign, success_mode=success_mode):
                total += p
        return total

    def test_examples_in_order(self):
        examples = [
            ("basic_tree.json", False),
            ("ref_example.json", False),
            ("connector.json", False),
            ("flat_tree.json", False),
            ("flat_tree_success.json", True),
            ("fta3.json", False),
            ("fta3_success.json", True),
            ("fta4.json", False),
            ("fta4_success.json", True),
            ("fta4b.json", False),
            ("large_tree.json", False),
            ("xor_simple.json", False),
            ("xor_simple.json", True),
            ("xor_big.json", False),
            ("xor_big.json", True),
            ("kn_simple.json", False),
            ("kn_simple.json", True),
            ("kn_big.json", False),
            ("kn_big.json", True),
            ("xor_kn_big.json", False),
            ("xor_kn_big.json", True),
            ("xor_kn_big_linear.json", False),
            ("xor_kn_big_linear.json", True),
        ]

        for filename, success_mode in examples:
            with self.subTest(filename=filename, success_mode=success_mode):
                actual, tree_norm, prob_input = self._run_example(filename, success_mode=success_mode)
                expected = self._bruteforce_probability(tree_norm, prob_input, success_mode=success_mode)
                if isinstance(expected, np.ndarray):
                    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
                else:
                    self.assertAlmostEqual(float(actual), float(expected), places=12)

    def test_flat_vs_cascade_match(self):
        prob_connector, _, _ = self._run_example("connector.json", success_mode=False)
        prob_flat, _, _ = self._run_example("flat_tree.json", success_mode=False)
        self.assertAlmostEqual(float(prob_connector), float(prob_flat), places=12)

if __name__ == '__main__':
    unittest.main()
