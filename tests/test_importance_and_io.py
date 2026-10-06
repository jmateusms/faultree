import builtins
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from faultree import analyze, minimal_cut_sets
from faultree.builder import load_probs_from_file, to_jsonable

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def leaf(event_id, prob):
    return {"id": event_id, "prob": prob}


def test_importance_measures_or_gate_match_closed_form():
    a, b = 0.1, 0.2
    result = analyze({"id": "TOP", "gate": "OR", "children": [leaf("A", a), leaf("B", b)]})
    q = 1 - (1 - a) * (1 - b)
    imp = result["importance"]["A"]
    assert imp["birnbaum"] == pytest.approx(1 - b)
    assert imp["criticality"] == pytest.approx((1 - b) * a / q)
    assert imp["raw"] == pytest.approx(1 / q)
    assert imp["rrw"] == pytest.approx(q / b)


def test_criticality_is_exact_risk_reduction_fraction():
    tree = {"id": "TOP", "gate": "OR", "children": [
        {"id": "G", "gate": "AND", "children": [leaf("A", 0.3), leaf("B", 0.4)]},
        {"id": "H", "gate": "K_OF_N", "k": 2, "children": [leaf("A", 0.3), leaf("C", 0.2), leaf("D", 0.1)]},
    ]}
    result = analyze(tree)
    for event_id, imp in result["importance"].items():
        q0 = result["conditional_Q"][event_id]["false"]
        assert imp["criticality"] == pytest.approx((result["Q"] - q0) / result["Q"])


def test_and_gate_rrw_is_infinite_and_serializes_as_null():
    result = analyze({"id": "TOP", "gate": "AND", "children": [leaf("A", 0.1), leaf("B", 0.2)]})
    imp = result["importance"]["A"]
    assert imp["criticality"] == pytest.approx(1.0)
    assert imp["raw"] == pytest.approx(1 / 0.1)
    assert math.isinf(imp["rrw"])
    assert to_jsonable(imp)["rrw"] is None
    json.dumps(to_jsonable(result), allow_nan=False)


def test_importance_is_elementwise_for_sample_vectors():
    tree = {"id": "TOP", "gate": "OR", "children": [leaf("A", [0.1, 0.5]), leaf("B", [0.2, 0.0])]}
    imp = analyze(tree)["importance"]["A"]
    np.testing.assert_allclose(imp["birnbaum"], [0.8, 1.0])
    np.testing.assert_allclose(imp["rrw"][0], 0.28 / 0.2)
    assert math.isinf(imp["rrw"][1])


def test_importance_not_defined_in_success_mode():
    tree = {"id": "TOP", "gate": "OR", "children": [leaf("A", 0.9), leaf("B", 0.8)]}
    assert analyze(tree, success_mode=True)["importance"] is None


def test_csv_probabilities_load_without_pandas(tmp_path, monkeypatch):
    path = tmp_path / "probs.csv"
    path.write_text("A,B\n0.1,0.2\n0.3,0.4\n")
    real_import = builtins.__import__

    def no_pandas(name, *args, **kwargs):
        if name == "pandas" or name.startswith("pandas."):
            raise ImportError("pandas blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pandas)
    probs = load_probs_from_file(str(path))
    np.testing.assert_allclose(probs["A"], [0.1, 0.3])
    np.testing.assert_allclose(probs["B"], [0.2, 0.4])
    with pytest.raises(ImportError, match=r"faultree\[excel\]"):
        load_probs_from_file(str(_touch(tmp_path / "probs.xlsx")))


def _touch(path):
    path.write_bytes(b"")
    return path


def test_csv_ragged_and_non_numeric_cells_are_reported(tmp_path):
    ragged = tmp_path / "ragged.csv"
    ragged.write_text("A,B\n0.1,0.2\n0.3\n")
    probs = load_probs_from_file(str(ragged))
    assert np.isnan(probs["B"][1])
    tree = {"id": "TOP", "gate": "OR", "children": [{"id": "A"}, {"id": "B"}]}
    with pytest.raises(ValueError, match="non-finite"):
        analyze(tree, str(ragged))
    bad = tmp_path / "bad.csv"
    bad.write_text("A\nabc\n")
    with pytest.raises(ValueError, match="non-numeric"):
        load_probs_from_file(str(bad))


def test_cli_cut_sets_prints_minimal_cuts():
    completed = subprocess.run(
        [sys.executable, "-m", "faultree", "examples/basic_tree.json", "--cut-sets"],
        capture_output=True, text=True, cwd=REPO, check=False)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    with open(os.path.join(REPO, "examples", "basic_tree.json")) as fh:
        expected = minimal_cut_sets(json.load(fh))
    assert payload["cut_sets"] == expected["cut_sets"]
    assert payload["complete"] is True
