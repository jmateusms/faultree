import builtins
from itertools import product
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
    dup = tmp_path / "dup.csv"
    dup.write_text("A,A\n0.1,0.2\n")
    with pytest.raises(ValueError, match="unique"):
        load_probs_from_file(str(dup))
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


def _fails(node, state):
    """Failure-mode structure function, written independently of the engine."""
    if "gate" not in node:
        return state[node["id"]]
    values = [_fails(child, state) for child in node["children"]]
    gate = node["gate"]
    if gate == "AND":
        return all(values)
    if gate == "OR":
        return any(values)
    if gate == "XOR":
        return sum(values) == 1
    if gate == "K_OF_N":
        return sum(values) >= node["k"]
    raise AssertionError(gate)


def _enumerated_q(tree, probs, forced=None):
    ids = sorted(probs)
    total = 0.0
    for values in product((False, True), repeat=len(ids)):
        state = dict(zip(ids, values))
        if forced is not None:
            state[forced[0]] = forced[1]
        weight = 1.0
        for event_id, value in zip(ids, values):
            if forced is not None and event_id == forced[0]:
                continue
            weight *= probs[event_id] if value else 1.0 - probs[event_id]
        if forced is not None and values[ids.index(forced[0])]:
            continue  # count each state of the other events once
        total += weight * _fails(tree, state)
    return total


@pytest.mark.parametrize("gate", ["AND", "K_OF_N", "XOR"])
def test_importance_matches_truth_table_with_repeated_events(gate):
    probs = {"A": 0.3, "B": 0.6, "C": 0.7, "D": 0.1}
    inner = {"id": "G2", "gate": gate, "children": [
        leaf("A", probs["A"]), leaf("C", probs["C"]), leaf("D", probs["D"])]}
    if gate == "K_OF_N":
        inner["k"] = 2
    tree = {"id": "TOP", "gate": "OR", "children": [
        {"id": "G1", "gate": "AND", "children": [leaf("A", probs["A"]), leaf("B", probs["B"])]},
        inner,
    ]}
    result = analyze(tree)
    q = _enumerated_q(tree, probs)
    assert result["Q"] == pytest.approx(q)
    for event_id, p in probs.items():
        q1 = _enumerated_q(tree, probs, (event_id, True))
        q0 = _enumerated_q(tree, probs, (event_id, False))
        imp = result["importance"][event_id]
        assert imp["birnbaum"] == pytest.approx(q1 - q0)
        assert imp["criticality"] == pytest.approx((q1 - q0) * p / q)
        assert imp["raw"] == pytest.approx(q1 / q)
        if q0 == 0:  # e.g. A guards every cut set when G2 is an AND gate
            assert math.isinf(imp["rrw"]) and imp["criticality"] == pytest.approx(1.0)
        else:
            assert imp["rrw"] == pytest.approx(q / q0)
            assert imp["criticality"] == pytest.approx(1 - 1 / imp["rrw"])
    if gate == "XOR":
        # Non-coherent: failing D can make the top event less likely.
        d = result["importance"]["D"]
        assert d["birnbaum"] < 0 and d["criticality"] < 0 and d["raw"] < 1


def test_csv_trailing_separators_are_ignored_but_unnamed_values_are_not(tmp_path):
    trailing = tmp_path / "trailing.csv"
    trailing.write_text("A,B,\n0.1,0.2,\n0.3,0.4,\n")
    probs = load_probs_from_file(str(trailing))
    assert sorted(probs) == ["A", "B"]
    np.testing.assert_allclose(probs["B"], [0.2, 0.4])
    unnamed = tmp_path / "unnamed.csv"
    unnamed.write_text("A,,B\n0.1,0.5,0.2\n")
    with pytest.raises(ValueError, match="no name"):
        load_probs_from_file(str(unnamed))


def test_cli_cut_sets_accepts_file_after_flag_and_rejects_reliability():
    def run(*args):
        return subprocess.run([sys.executable, "-m", "faultree", *args],
                              capture_output=True, text=True, cwd=REPO, check=False)

    before = run("--cut-sets", "examples/basic_tree.json")
    assert before.returncode == 0, before.stderr
    assert json.loads(before.stdout)["limits"]["max_order"] == 6
    ordered = run("--cut-sets", "1", "examples/basic_tree.json")
    assert ordered.returncode == 0, ordered.stderr
    assert json.loads(ordered.stdout)["cut_sets"] == []
    rejected = run("examples/basic_tree.json", "--cut-sets", "--reliability")
    assert rejected.returncode == 2
    assert "--reliability" in rejected.stderr


def test_cli_reports_missing_excel_extra_without_traceback(tmp_path, monkeypatch, capsys):
    from faultree import cli

    excel = _touch(tmp_path / "probs.xlsx")
    real_import = builtins.__import__

    def no_pandas(name, *args, **kwargs):
        if name == "pandas" or name.startswith("pandas."):
            raise ImportError("pandas blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pandas)
    monkeypatch.setattr(sys, "argv", [
        "faultree", os.path.join(REPO, "examples", "basic_tree.json"), "--probs", str(excel)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert "faultree[excel]" in str(exc.value.code)


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_csv_with_semicolons_and_decimal_commas(tmp_path):
    # The default CSV of a pt-BR Excel: ';' between columns, ',' for decimals.
    path = _write(tmp_path, "probs.csv", "A;B;C\n0,1;0,25;1e-3\n0,2;0,5;2,5E-4\n")
    columns = load_probs_from_file(path)
    np.testing.assert_allclose(columns["A"], [0.1, 0.2])
    np.testing.assert_allclose(columns["B"], [0.25, 0.5])
    np.testing.assert_allclose(columns["C"], [1e-3, 2.5e-4])
    tree = {"id": "TOP", "gate": "OR", "children": [leaf("A", None), leaf("B", None), leaf("C", None)]}
    for node in tree["children"]:
        del node["prob"]
    np.testing.assert_allclose(analyze(tree, path)["Q"], 1 - (1 - columns["A"]) * (1 - columns["B"]) * (1 - columns["C"]))


def test_csv_with_tabs_and_a_single_decimal_comma_column(tmp_path):
    tabs = load_probs_from_file(_write(tmp_path, "tabs.csv", "A\tB\n0,1\t0.2\n"))
    assert tabs["A"].tolist() == [0.1] and tabs["B"].tolist() == [0.2]
    single = load_probs_from_file(_write(tmp_path, "one.csv", "A\n0,05\n0,07\n"))
    np.testing.assert_allclose(single["A"], [0.05, 0.07])


def test_comma_csv_keeps_its_rules_and_hints_at_decimal_commas(tmp_path):
    plain = load_probs_from_file(_write(tmp_path, "plain.csv", "A,B\n0.1,0.2\n"))
    assert plain["A"].tolist() == [0.1] and plain["B"].tolist() == [0.2]
    with pytest.raises(ValueError, match="decimal commas"):
        load_probs_from_file(_write(tmp_path, "mixed.csv", "A,B\n0,1,0,2\n"))
    with pytest.raises(ValueError, match="non-numeric"):
        load_probs_from_file(_write(tmp_path, "bad.csv", "A;B\n0,1;x\n"))


def test_probability_file_extension_is_case_insensitive(tmp_path):
    columns = load_probs_from_file(_write(tmp_path, "PROBS.CSV", "A,B\n0.1,0.2\n"))
    assert columns["A"].tolist() == [0.1]
    with pytest.raises(ValueError, match="Unsupported"):
        load_probs_from_file(_write(tmp_path, "probs.txt", "A\n0.1\n"))

