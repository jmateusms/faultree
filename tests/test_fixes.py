"""Regression tests for the 0.3.0 engine/API fixes.

Each test class maps to a finding from the Phase 0 review:
A-1 exponential quantification, A-2 success-mode XOR dual, A-3 silent
missing probabilities, A-4 cycle detection, A-5 unseeded resampling,
input validation (ordering / empty gates / k > n / probability range),
and P-1/P-2 server behavior.
"""
import json
import os
import time
import unittest

import numpy as np

from faultree.builder import (
    align_probabilities,
    build,
    compute_event_probabilities,
    normalize_tree,
    wmc,
)

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), '..', 'examples')


def _top_prob(tree, success_mode=False, probs=None, **kwargs):
    bdd, _, _, _ = build(tree, success_mode=success_mode)
    prob_map = compute_event_probabilities(
        bdd, tree, probs, success_mode=success_mode, **kwargs)
    return prob_map[str(normalize_tree(tree)["id"])]


def _complement_probs(node):
    """Return a copy of the tree with every leaf prob q replaced by 1 - q."""
    out = dict(node)
    if "prob" in out and out["prob"] is not None:
        out["prob"] = 1.0 - out["prob"]
    if out.get("children"):
        out["children"] = [_complement_probs(ch) for ch in out["children"]]
    return out


class TestQuantificationIsPolynomial(unittest.TestCase):
    """A-1: wmc must be O(|BDD|), not exponential in satisfying cubes."""

    def test_parity_20_vars_fast_and_exact(self):
        from dd.autoref import BDD
        n = 20
        bdd = BDD()
        names = [f"x{i}" for i in range(n)]
        bdd.declare(*names)
        f = bdd.add_expr(" ^ ".join(names))
        start = time.perf_counter()
        val = wmc(bdd, f, {k: 0.5 for k in names})
        elapsed = time.perf_counter() - start
        self.assertAlmostEqual(val, 0.5, places=12)
        # Pre-fix this took ~4 s (524288 cubes); the cofactor recursion
        # visits each of the ~211 nodes once. Generous bound for slow CI.
        self.assertLess(elapsed, 2.0)

    def test_complemented_edges_handled(self):
        # ~(x & y): the top reference is complemented in dd.autoref; a wrong
        # sign convention would silently return 1 - Q.
        from dd.autoref import BDD
        bdd = BDD()
        bdd.declare('x', 'y')
        f = bdd.add_expr('~ (x & y)')
        val = wmc(bdd, f, {'x': 0.1, 'y': 0.2})
        self.assertAlmostEqual(val, 1.0 - 0.1 * 0.2, places=12)

    def test_array_probabilities_broadcast(self):
        tree = {"id": "T", "gate": "OR", "children": [
            {"id": "a", "prob": [0.1, 0.2]}, {"id": "b", "prob": [0.3, 0.4]}]}
        top = _top_prob(tree)
        expected = 1.0 - (1.0 - np.array([0.1, 0.2])) * (1.0 - np.array([0.3, 0.4]))
        np.testing.assert_allclose(top, expected, rtol=1e-12)

    def test_rare_event_relative_accuracy(self):
        # Complement edges must be pushed into the children, never computed
        # as 1 - r at the reference: that floors rare-event results at the
        # ~1e-16 absolute noise level (this tree returned 0.0 for 3e-18
        # under the naive complement handling).
        p = 1e-9
        tree = {"id": "T", "gate": "AND", "children": [
            {"id": "X", "gate": "XOR", "children": [
                {"id": "a", "prob": p}, {"id": "b", "prob": p},
                {"id": "c", "prob": p}]},
            {"id": "d", "prob": p}]}
        exact = 3.0 * p * (1.0 - p) ** 2 * p  # P(exactly one of 3) * P(d)
        top = float(_top_prob(tree))
        self.assertGreater(top, 0.0)
        self.assertLess(abs(top - exact) / exact, 1e-12)


class TestSuccessModeDual(unittest.TestCase):
    """A-2: success mode must satisfy R = 1 - Q for every gate type."""

    def _assert_parity(self, fail_tree):
        q_fail = _top_prob(fail_tree, success_mode=False)
        success_tree = _complement_probs(fail_tree)
        r_success = _top_prob(success_tree, success_mode=True)
        self.assertAlmostEqual(float(r_success), 1.0 - float(q_fail), places=12)

    def test_xor_n2(self):
        self._assert_parity({"id": "T", "gate": "XOR", "children": [
            {"id": "a", "prob": 0.1}, {"id": "b", "prob": 0.2}]})

    def test_xor_n3(self):
        self._assert_parity({"id": "T", "gate": "XOR", "children": [
            {"id": "a", "prob": 0.1}, {"id": "b", "prob": 0.2},
            {"id": "c", "prob": 0.3}]})

    def test_xor_n4(self):
        self._assert_parity({"id": "T", "gate": "XOR", "children": [
            {"id": "a", "prob": 0.1}, {"id": "b", "prob": 0.2},
            {"id": "c", "prob": 0.3}, {"id": "d", "prob": 0.4}]})

    def test_xor_n3_known_values(self):
        # The exact numbers from the review: Q_fail = 0.398, so R = 0.602
        # (the pre-fix engine returned 0.908).
        fail = {"id": "T", "gate": "XOR", "children": [
            {"id": "a", "prob": 0.1}, {"id": "b", "prob": 0.2},
            {"id": "c", "prob": 0.3}]}
        self.assertAlmostEqual(float(_top_prob(fail)), 0.398, places=12)
        succ = _complement_probs(fail)
        self.assertAlmostEqual(
            float(_top_prob(succ, success_mode=True)), 0.602, places=12)

    def test_k_of_n(self):
        self._assert_parity({"id": "T", "gate": "K_OF_N", "k": 2, "children": [
            {"id": "a", "prob": 0.1}, {"id": "b", "prob": 0.2},
            {"id": "c", "prob": 0.3}]})

    def test_mixed_tree(self):
        self._assert_parity({"id": "T", "gate": "OR", "children": [
            {"id": "G1", "gate": "XOR", "children": [
                {"id": "a", "prob": 0.1}, {"id": "b", "prob": 0.2},
                {"id": "c", "prob": 0.3}]},
            {"id": "G2", "gate": "AND", "children": [
                {"id": "d", "prob": 0.4}, {"id": "e", "prob": 0.5}]}]})

    def test_tree_with_ref_nodes(self):
        with open(os.path.join(EXAMPLES_DIR, "ref_example.json")) as fh:
            self._assert_parity(json.load(fh))


class TestMissingProbability(unittest.TestCase):
    """A-3: a basic event without a probability must raise, not default to 0."""

    TREE = {"id": "T", "gate": "OR", "children": [
        {"id": "a", "prob": 0.1}, {"id": "b"}]}

    def test_missing_raises(self):
        with self.assertRaisesRegex(ValueError, "No probability defined.*b"):
            _top_prob(self.TREE)

    def test_explicit_optout(self):
        self.assertAlmostEqual(
            float(_top_prob(self.TREE, allow_missing=True)), 0.1, places=12)

    def test_optout_with_array_sibling(self):
        tree = {"id": "T", "gate": "OR", "children": [
            {"id": "a", "prob": [0.1, 0.2]}, {"id": "b"}]}
        top = _top_prob(tree, allow_missing=True)
        np.testing.assert_allclose(top, [0.1, 0.2], rtol=1e-12)

    def test_out_of_range_raises(self):
        tree = {"id": "T", "gate": "OR", "children": [{"id": "a", "prob": 1.5}]}
        with self.assertRaisesRegex(ValueError, r"outside \[0, 1\].*a"):
            _top_prob(tree)

    def test_non_numeric_prob_raises(self):
        tree = {"id": "T", "gate": "OR", "children": [{"id": "a", "prob": "abc"}]}
        with self.assertRaisesRegex(ValueError, "non-numeric.*a"):
            _top_prob(tree)


class TestCycleDetection(unittest.TestCase):
    """A-4: cyclic ref / branch structures must raise, not recurse forever."""

    def test_ref_cycle(self):
        tree = {"id": "A", "gate": "AND", "children": [
            {"id": "B", "gate": "OR", "children": [{"ref": "A"}]},
            {"id": "c", "prob": 0.5}]}
        with self.assertRaisesRegex(ValueError, "Cycle detected.*A -> B -> A"):
            build(tree)

    def test_flat_branch_cycle(self):
        flat = {"ft_nodes": [
            {"label": "G1", "gate": "AND", "branches": ["G2"]},
            {"label": "G2", "gate": "OR", "branches": ["G1"]}],
            "be_nodes": []}
        with self.assertRaisesRegex(ValueError, "Cycle detected"):
            normalize_tree(flat)

    def test_dag_sharing_is_not_a_cycle(self):
        # The same subtree referenced from two places is legal DAG sharing.
        tree = {"id": "T", "gate": "OR", "children": [
            {"id": "G1", "gate": "AND", "children": [
                {"id": "shared", "prob": 0.1}, {"id": "b", "prob": 0.2}]},
            {"id": "G2", "gate": "AND", "children": [
                {"ref": "shared"}, {"id": "c", "prob": 0.3}]}]}
        self.assertGreater(float(_top_prob(tree)), 0.0)

    def test_duplicate_id_on_path_rejected(self):
        # Old code silently clobbered the root's registry entry and reported
        # the leaf's probability as the top event.
        tree = {"id": "T", "gate": "OR", "children": [
            {"id": "T", "prob": 0.25}, {"id": "b", "prob": 0.5}]}
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            build(tree)


class TestInputValidation(unittest.TestCase):

    def test_incomplete_ordering_raises(self):
        tree = {"id": "T", "gate": "OR", "children": [
            {"id": "a", "prob": 0.1}, {"id": "b", "prob": 0.2}]}
        with self.assertRaisesRegex(ValueError, "missing: b"):
            build(tree, ordering=["a"])

    def test_duplicate_ordering_raises(self):
        tree = {"id": "T", "gate": "OR", "children": [
            {"id": "a", "prob": 0.1}, {"id": "b", "prob": 0.2}]}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build(tree, ordering=["a", "b", "a"])

    def test_complete_ordering_matches_default(self):
        tree = {"id": "T", "gate": "OR", "children": [
            {"id": "a", "prob": 0.1}, {"id": "b", "prob": 0.2}]}
        default = _top_prob(tree)
        bdd, _, _, _ = build(tree, ordering=["b", "a"])
        reordered = compute_event_probabilities(bdd, tree, None)["T"]
        self.assertAlmostEqual(float(default), float(reordered), places=12)

    def test_empty_gate_raises(self):
        with self.assertRaisesRegex(ValueError, "has no children"):
            build({"id": "T", "gate": "AND", "children": []})

    def test_k_greater_than_n_raises(self):
        tree = {"id": "T", "gate": "K_OF_N", "k": 3, "children": [
            {"id": "a", "prob": 0.1}, {"id": "b", "prob": 0.2}]}
        with self.assertRaisesRegex(ValueError, "k=3 but only 2 children"):
            build(tree)

    def test_non_integer_k_raises(self):
        tree = {"id": "T", "gate": "K_OF_N", "k": "2", "children": [
            {"id": "a", "prob": 0.1}, {"id": "b", "prob": 0.2}]}
        with self.assertRaisesRegex(ValueError, "integer field 'k'"):
            build(tree)

    def test_bool_k_raises(self):
        # bool subclasses int; k=True must not be accepted as k=1.
        tree = {"id": "T", "gate": "K_OF_N", "k": True, "children": [
            {"id": "a", "prob": 0.1}, {"id": "b", "prob": 0.2}]}
        with self.assertRaisesRegex(ValueError, "integer field 'k'"):
            build(tree)

    def test_leaf_without_id_raises(self):
        tree = {"id": "T", "gate": "OR", "children": [
            {"prob": 0.5}, {"id": "b", "prob": 0.1}]}
        with self.assertRaisesRegex(ValueError, "no 'id' field"):
            build(tree)

    def test_children_wrong_type_raises(self):
        tree = {"id": "T", "gate": "OR", "children": "oops"}
        with self.assertRaisesRegex(ValueError, "must be a list"):
            build(tree)

    def test_flat_node_without_label_raises(self):
        flat = {"ft_nodes": [{"gate": "OR", "branches": ["a"]}],
                "be_nodes": [{"label": "a", "prob": 0.1}]}
        with self.assertRaisesRegex(ValueError, "'label'"):
            normalize_tree(flat)


class TestSampleVectorContracts(unittest.TestCase):
    """Sample vectors are aligned joint samples unless explicitly resampled."""

    def _ragged(self):
        return {"a": np.array([0.1, 0.2, 0.3, 0.4]), "b": np.array([0.5, 0.6])}

    def test_same_seed_same_result(self):
        r1 = align_probabilities(self._ragged(), seed=0, resample_independent=True)
        r2 = align_probabilities(self._ragged(), seed=0, resample_independent=True)
        np.testing.assert_array_equal(r1["b"], r2["b"])

    def test_shuffle_reproducible(self):
        r1 = align_probabilities(self._ragged(), shuffle=True, seed=42,
                                 resample_independent=True)
        r2 = align_probabilities(self._ragged(), shuffle=True, seed=42,
                                 resample_independent=True)
        np.testing.assert_array_equal(r1["a"], r2["a"])
        np.testing.assert_array_equal(r1["b"], r2["b"])

    def test_resampled_length(self):
        r = align_probabilities(self._ragged(), seed=0, resample_independent=True)
        self.assertEqual(len(r["b"]), 4)

    def test_unequal_vectors_rejected_without_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, "equal lengths.*resample_independent"):
            align_probabilities(self._ragged())

    def test_shared_shuffle_preserves_joint_pairs(self):
        probs = {"a": np.array([0.1, 0.2, 0.3]),
                 "b": np.array([0.4, 0.5, 0.6])}
        shuffled = align_probabilities(probs, shuffle=True, seed=42)
        np.testing.assert_allclose(shuffled["b"] - shuffled["a"], 0.3)

    def test_empty_or_multidimensional_vectors_rejected(self):
        for prob in ([], [[0.1, 0.2]]):
            tree = {"id": "T", "gate": "OR", "children": [
                {"id": "a", "prob": prob}]}
            with self.subTest(prob=prob):
                with self.assertRaisesRegex(ValueError, "finite scalars or non-empty"):
                    _top_prob(tree)


class TestEventIdentityAndFlatRoot(unittest.TestCase):

    def test_conflicting_recursive_duplicate_id_rejected(self):
        tree = {"id": "T", "gate": "OR", "children": [
            {"id": "a", "prob": 0.1}, {"id": "a", "prob": 0.2}]}
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate id 'a'"):
            build(tree)

    def test_identical_recursive_duplicate_id_is_shared_event(self):
        tree = {"id": "T", "gate": "AND", "children": [
            {"id": "a", "prob": 0.2}, {"id": "a", "prob": 0.2}]}
        self.assertAlmostEqual(float(_top_prob(tree)), 0.2, places=12)

    def test_conflicting_flat_duplicate_id_rejected(self):
        flat = {"ft_nodes": [
            {"label": "T", "gate": "OR", "branches": ["a"]},
            {"label": "T", "gate": "AND", "branches": ["a"]}],
            "be_nodes": [{"label": "a", "prob": 0.1}]}
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate id 'T'"):
            normalize_tree(flat)

    def test_ambiguous_flat_root_rejected(self):
        flat = {"ft_nodes": [
            {"label": "T1", "gate": "OR", "branches": ["a"]},
            {"label": "T2", "gate": "OR", "branches": ["b"]}],
            "be_nodes": [{"label": "a", "prob": 0.1}, {"label": "b", "prob": 0.2}]}
        with self.assertRaisesRegex(ValueError, "exactly one unreferenced root"):
            normalize_tree(flat)

    def test_disconnected_flat_node_rejected_with_explicit_root(self):
        flat = {"analysis": {"initiator": "I"},
            "esd_nodes": [{"label": "I", "ft": "T"}],
            "ft_nodes": [
                {"label": "T", "gate": "OR", "branches": ["a"]},
                {"label": "unused", "gate": "OR", "branches": ["b"]}],
            "be_nodes": [{"label": "a", "prob": 0.1}, {"label": "b", "prob": 0.2}]}
        with self.assertRaisesRegex(ValueError, "disconnected/inaccessible.*b.*unused"):
            normalize_tree(flat)


class TestServer(unittest.TestCase):
    """P-1: flat-format trees must return a real top_event_probability.
    P-2: array probabilities must serialize to JSON."""

    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient
            from faultree.server import app
        except ImportError:
            raise unittest.SkipTest("fastapi/httpx not installed")
        cls.client = TestClient(app)

    def test_flat_format_top_event(self):
        with open(os.path.join(EXAMPLES_DIR, "flat_tree.json")) as fh:
            flat = json.load(fh)
        expected = float(_top_prob(flat))
        resp = self.client.post("/analyze", json={"tree": flat})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIsNotNone(body["top_event_probability"])
        self.assertAlmostEqual(body["top_event_probability"], expected, places=12)

    def test_array_probs_serialize(self):
        tree = {"id": "T", "gate": "OR", "children": [
            {"id": "a", "prob": [0.1, 0.2]}, {"id": "b", "prob": [0.3, 0.4]}]}
        resp = self.client.post("/analyze", json={"tree": tree})
        self.assertEqual(resp.status_code, 200)
        top = resp.json()["top_event_probability"]
        self.assertIsInstance(top, list)
        expected = 1.0 - (1.0 - np.array([0.1, 0.2])) * (1.0 - np.array([0.3, 0.4]))
        np.testing.assert_allclose(top, expected, rtol=1e-12)

    def test_invalid_tree_is_400_not_500(self):
        tree = {"id": "T", "gate": "OR", "children": [{"id": "a"}]}
        resp = self.client.post("/analyze", json={"tree": tree})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("No probability defined", resp.json()["detail"])

    def test_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")

    def test_reliability_flag_applies(self):
        # End-to-end wiring of the reliability field: XOR n=3 success mode
        # must give 0.602, not the pre-fix 0.908.
        tree = {"id": "T", "gate": "XOR", "children": [
            {"id": "a", "prob": 0.9}, {"id": "b", "prob": 0.8},
            {"id": "c", "prob": 0.7}]}
        resp = self.client.post(
            "/analyze", json={"tree": tree, "reliability": True})
        self.assertEqual(resp.status_code, 200)
        self.assertAlmostEqual(
            resp.json()["top_event_probability"], 0.602, places=12)

    def test_malformed_leaf_is_400(self):
        tree = {"id": "T", "gate": "OR", "children": [
            {"prob": 0.5}, {"id": "b", "prob": 0.1}]}
        resp = self.client.post("/analyze", json={"tree": tree})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("id", resp.json()["detail"])

    def test_children_wrong_type_is_400(self):
        resp = self.client.post(
            "/analyze",
            json={"tree": {"id": "T", "gate": "OR", "children": "oops"}})
        self.assertEqual(resp.status_code, 400)

    def test_unequal_sample_vectors_require_explicit_opt_in(self):
        tree = {"id": "T", "gate": "OR", "children": [
            {"id": "a", "prob": [0.1, 0.2]}, {"id": "b", "prob": [0.3]}]}
        rejected = self.client.post("/analyze", json={"tree": tree})
        self.assertEqual(rejected.status_code, 400)
        self.assertIn("equal lengths", rejected.json()["detail"])
        accepted = self.client.post(
            "/analyze", json={"tree": tree, "resample_independent": True})
        self.assertEqual(accepted.status_code, 200)


class TestCLI(unittest.TestCase):
    """End-to-end CLI coverage (subprocess, same interpreter)."""

    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _run(self, *args):
        import subprocess
        import sys
        return subprocess.run(
            [sys.executable, "-m", "faultree", *args],
            capture_output=True, text=True, cwd=self.REPO)

    def test_example_runs(self):
        r = self._run("examples/basic_tree.json")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Top event probability [TE]: 0.018444", r.stdout)

    def test_missing_file_friendly_error(self):
        r = self._run("does_not_exist.json")
        self.assertEqual(r.returncode, 1)
        self.assertIn("Error: cannot read", r.stderr)
        self.assertNotIn("Traceback", r.stderr)

    def test_bad_probs_json_friendly_error(self):
        r = self._run("examples/basic_tree.json", "--probs", "{bad")
        self.assertEqual(r.returncode, 1)
        self.assertIn("failed to parse", r.stderr)
        self.assertNotIn("Traceback", r.stderr)

    def test_missing_prob_error_and_optout(self):
        import tempfile
        tree = {"id": "T", "gate": "OR", "children": [
            {"id": "a", "prob": 0.1}, {"id": "b"}]}
        with tempfile.NamedTemporaryFile(
                "w", suffix=".json", delete=False) as fh:
            json.dump(tree, fh)
            path = fh.name
        try:
            r = self._run(path)
            self.assertEqual(r.returncode, 1)
            self.assertIn("No probability defined", r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            r2 = self._run(path, "--assume-missing-zero")
            self.assertEqual(r2.returncode, 0, r2.stderr)
            self.assertIn("Top event probability [T]: 0.1", r2.stdout)
        finally:
            os.unlink(path)

    def test_unequal_sample_vectors_require_explicit_opt_in(self):
        import tempfile
        tree = {"id": "T", "gate": "OR", "children": [
            {"id": "a", "prob": [0.1, 0.2]}, {"id": "b", "prob": [0.3]}]}
        with tempfile.NamedTemporaryFile(
                "w", suffix=".json", delete=False) as fh:
            json.dump(tree, fh)
            path = fh.name
        try:
            rejected = self._run(path)
            self.assertEqual(rejected.returncode, 1)
            self.assertIn("equal lengths", rejected.stderr)
            accepted = self._run(path, "--resample-independent")
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
        finally:
            os.unlink(path)


if __name__ == '__main__':
    unittest.main()


def test_duplicate_error_reports_both_positions():
    import pytest
    from faultree.builder import normalize_tree
    tree = {"id": "T", "gate": "OR", "children": [
        {"id": "A", "prob": .1}, {"id": "A", "prob": .2}]}
    with pytest.raises(ValueError, match=r"root.children\[0\].*root.children\[1\]"):
        normalize_tree(tree)


def test_explicit_independent_resampling_covers_equal_length_marginals():
    from faultree.builder import align_probabilities
    values = np.arange(100) / 100
    aligned = align_probabilities({"a": values, "b": values}, resample_independent=True, seed=1)
    assert not np.array_equal(aligned["a"], aligned["b"])
