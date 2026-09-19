"""Structured-result and reusable-compilation regression coverage."""
from itertools import product
import json
import os
import subprocess
import sys

import pytest

from faultree import analyze, benchmark_orderings, compile_tree, quantify_compiled


XOR_TREE = {
    "id": "T",
    "gate": "XOR",
    "children": [
        {"id": "a", "prob": 0.2},
        {"id": "b", "prob": 0.3},
        {"id": "c", "prob": 0.4},
    ],
}


def _xor_probability(probs, forced=None):
    """Independent exhaustive oracle for exactly-one of the three leaves."""
    total = 0.0
    ids = tuple(probs)
    for values in product((False, True), repeat=len(ids)):
        state = dict(zip(ids, values))
        if forced and state[forced[0]] is not forced[1]:
            continue
        weight = 1.0
        for event_id, value in state.items():
            weight *= probs[event_id] if value else 1.0 - probs[event_id]
        if sum(values) == 1:
            total += weight
    if forced:
        # The loop above computes P(T and forced); divide by P(forced) to get
        # the conditioned result independently of the BDD implementation.
        event_id, value = forced
        total /= probs[event_id] if value else 1.0 - probs[event_id]
    return total


def test_structured_q_conditionals_and_signed_birnbaum_match_enumeration():
    probabilities = {"a": 0.2, "b": 0.3, "c": 0.4}
    result = analyze(XOR_TREE, ordering=["c", "b", "a"], seed=17)

    assert result["Q"] == pytest.approx(_xor_probability(probabilities))
    for event_id in probabilities:
        expected_true = _xor_probability(probabilities, (event_id, True))
        expected_false = _xor_probability(probabilities, (event_id, False))
        conditional = result["conditional_Q"][event_id]
        assert conditional["true"] == pytest.approx(expected_true)
        assert conditional["false"] == pytest.approx(expected_false)
        assert result["birnbaum"][event_id] == pytest.approx(
            expected_true - expected_false)

    # XOR is non-monotonic: failing ``a`` can make the exactly-one top event
    # less likely, so preserving a negative Birnbaum sign is material.
    assert result["birnbaum"]["a"] == pytest.approx(-0.04)
    assumptions = result["assumptions"]
    assert assumptions["basic_event_independence"] is True
    assert assumptions["variable_order"] == ["c", "b", "a"]
    assert assumptions["seed"] == 17
    assert assumptions["faultree_version"]


def test_compilation_is_reusable_and_benchmark_only_reports_caller_orders():
    compiled = compile_tree(XOR_TREE, ordering=["a", "b", "c"])
    first = quantify_compiled(compiled)["T"]
    second = quantify_compiled(compiled, {"a": 0.5, "b": 0.3, "c": 0.4})["T"]
    assert first != second
    assert compiled.ordering == ("a", "b", "c")

    benchmark = benchmark_orderings(XOR_TREE, [["a", "b", "c"], ["c", "b", "a"]])
    assert [row["ordering"] for row in benchmark] == [["a", "b", "c"], ["c", "b", "a"]]
    assert all(row["bdd_nodes"] > 0 and row["top_bdd_nodes"] > 0 for row in benchmark)
    assert all(row["compile_seconds"] >= 0.0 for row in benchmark)


def test_cli_structured_output_is_machine_readable_json():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    completed = subprocess.run(
        [sys.executable, "-m", "faultree", "examples/basic_tree.json", "--structured"],
        capture_output=True, text=True, cwd=repo, check=False)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["Q"] == pytest.approx(0.018444)
    assert payload["conditional_Q"]["BE1"]["true"] == pytest.approx(0.1537)


def test_api_keeps_legacy_fields_and_adds_structured_result():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from faultree.server import app

    response = TestClient(app).post("/analyze", json={"tree": XOR_TREE, "seed": 9})
    assert response.status_code == 200
    payload = response.json()
    assert payload["top_event_probability"] == pytest.approx(payload["result"]["Q"])
    assert payload["result"]["conditional_Q"]["a"]["true"] == pytest.approx(0.42)
    assert payload["result"]["birnbaum"]["a"] == pytest.approx(-0.04)


def test_benchmark_ordering_preserves_joint_scenario_probabilities():
    import numpy as np
    samples = {"a": [0.1, 0.8], "b": [0.2, 0.3], "c": [0.6, 0.1]}
    rows = benchmark_orderings(XOR_TREE, [["a", "b", "c"], ["c", "b", "a"]], probs_by_id=samples)
    np.testing.assert_allclose(rows[0]["Q"], rows[1]["Q"])
    assert all(row["sample_count"] == 2 and row["evaluation_seconds"] >= 0 for row in rows)
    assert analyze(XOR_TREE, samples)["assumptions"]["sample_count"] == 2


def test_compilation_can_skip_rendering_and_owns_its_source_snapshot(monkeypatch):
    from copy import deepcopy
    import faultree.builder as builder
    tree = deepcopy(XOR_TREE)
    def forbid(*args, **kwargs):
        raise AssertionError("expression rendering was requested")
    monkeypatch.setattr(builder, "node_to_expr", forbid)
    monkeypatch.setattr(builder, "node_to_symbolic", forbid)
    compiled = compile_tree(tree, include_expressions=False)
    original = quantify_compiled(compiled)["T"]
    tree["children"][0]["prob"] = 1.0
    assert quantify_compiled(compiled)["T"] == original
    assert compiled.expression == compiled.symbolic == ""
