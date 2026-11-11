"""The one-pass conditional Q of analyze() against the per-event path it
replaced (one restriction and one model count per event and value), on the
property-test trees, with scalar and sample-vector probabilities, and on
every example file."""
import copy
import json
import math
import os
import random

import numpy as np
import pytest

from faultree import analyze, builder

from .test_property import CASES, EXAMPLES, ORACLE_EXAMPLES, PROB_FILES, leaves, with_probs

REL = 1e-12


def both_paths(monkeypatch, model, *args, **kwargs):
    results = []
    for one_pass in (True, False):
        monkeypatch.setattr(builder, "_ONE_PASS_IMPORTANCE", one_pass)
        results.append(analyze(copy.deepcopy(model), *args, **kwargs))
    return results


def assert_close(new, old, floor, where):
    """Same type and shape; values within REL relative, or within ``floor``
    of each other; inf and NaN in the same places."""
    assert type(new) is type(old), (where, new, old)
    a, b = np.asarray(new, dtype=float), np.asarray(old, dtype=float)
    assert a.shape == b.shape, (where, new, old)
    assert np.array_equal(np.isnan(a), np.isnan(b)), (where, new, old)
    assert np.array_equal(np.isinf(a), np.isinf(b)) and np.array_equal(a[np.isinf(a)], b[np.isinf(b)]), (where, new, old)
    finite = np.isfinite(a)
    gap = np.abs(a[finite] - b[finite])
    bound = REL * np.maximum(np.abs(a[finite]), np.abs(b[finite])) + np.broadcast_to(floor, a.shape)[finite]
    assert np.all(gap <= bound), (where, new, old, float(np.max(gap / np.maximum(bound, 1e-300))))


def compare(new, old, where):
    """Conditional Q, RAW and RRW to REL relative with no floor: they are
    sums or ratios of sums of non-negative terms. Birnbaum and criticality
    are differences, Q1 - Q0 and (Q1 - Q0) q / Q, which cannot be more
    accurate than their operands: they get an absolute floor of REL times
    the larger of Q1 and Q0 (times q / Q for criticality)."""
    assert new["conditional_Q"].keys() == old["conditional_Q"].keys()
    assert new["Q"] is old["Q"] or np.array_equal(new["Q"], old["Q"])
    for event, values in old["conditional_Q"].items():
        for key in ("true", "false"):
            assert_close(new["conditional_Q"][event][key], values[key], 0.0, (where, event, key))
            assert np.all(np.asarray(new["conditional_Q"][event][key]) <= 1.0), (where, event, key)
        scale = np.maximum(np.abs(np.asarray(values["true"], dtype=float)),
                           np.abs(np.asarray(values["false"], dtype=float)))
        assert_close(new["birnbaum"][event], old["birnbaum"][event], REL * scale, (where, event, "birnbaum"))
        if old["importance"] is None:
            assert new["importance"] is None
            continue
        assert new["importance"][event].keys() == old["importance"][event].keys()
        with np.errstate(divide="ignore", invalid="ignore"):
            q_over_top = np.asarray(old["probabilities"][event], dtype=float) / np.asarray(old["Q"], dtype=float)
            floors = (("birnbaum", REL * scale), ("criticality", REL * scale * q_over_top),
                      ("raw", 0.0), ("rrw", 0.0))
        for measure, floor in floors:
            assert_close(new["importance"][event][measure], old["importance"][event][measure],
                         floor, (where, event, measure))


def vector_case(rng, tree, ids, size=4):
    """``tree`` with sample vectors for some or all events (mixed scalar and
    vector inputs reach the per-event recomputation of the one-pass path)."""
    probs = {}
    for event in ids:
        if rng.random() < 0.6:
            probs[event] = [rng.choice((0.0, 1.0, 1e-9, rng.random())) if rng.random() < 0.2
                            else rng.random() for _ in range(size)]
        else:
            probs[event] = rng.choice((0.0, 1.0, 1e-9, rng.random()))
    return with_probs(tree, probs)


@pytest.mark.parametrize("success_mode", [False, True])
def test_property_trees_agree(monkeypatch, success_mode):
    for name, (tree, _, _) in CASES:
        new, old = both_paths(monkeypatch, tree, success_mode=success_mode)
        compare(new, old, (name, success_mode))


@pytest.mark.parametrize("success_mode", [False, True])
def test_property_trees_with_sample_vectors_agree(monkeypatch, success_mode):
    rng = random.Random(42)
    for name, (tree, ids, _) in CASES[::2]:
        model = vector_case(rng, tree, ids)
        new, old = both_paths(monkeypatch, model, success_mode=success_mode)
        compare(new, old, (name, success_mode))


@pytest.mark.parametrize("name,success_mode",
                         [(n, m) for n, modes in ORACLE_EXAMPLES.items() for m in modes])
def test_examples_agree(monkeypatch, name, success_mode):
    with open(os.path.join(EXAMPLES, name), encoding="utf-8") as fh:
        model = json.load(fh)
    prob_file = PROB_FILES.get(name) or model.get("prob_file")
    prob_input = os.path.join(EXAMPLES, prob_file) if prob_file else None
    new, old = both_paths(monkeypatch, model, prob_input, success_mode=success_mode)
    compare(new, old, (name, success_mode))


def test_missing_and_absent_events_and_constant_restrictions(monkeypatch):
    # B has no probability (allow_missing), H switches C out of the BDD, D
    # forced true makes the top certain, E is a declared variable outside
    # the top event's BDD.
    tree = {"id": "T", "gate": "OR", "children": [
        {"id": "G", "gate": "AND", "children": [
            {"id": "A", "prob": 0.3}, {"id": "B"},
            {"id": "G2", "gate": "AND", "children": [
                {"id": "H", "event_type": "house", "state": False}, {"id": "C", "prob": 0.2}]}]},
        {"id": "D", "prob": [0.1, 0.2, 0.3]}]}
    new, old = both_paths(monkeypatch, tree, allow_missing=True,
                          ordering=["E", "A", "B", "C", "D"])
    compare(new, old, "missing")
    assert new["conditional_Q"]["D"]["true"] == 1.0 and type(new["conditional_Q"]["D"]["true"]) is float
    assert new["conditional_Q"]["C"]["true"] is new["Q"]


def test_sample_vectors_on_every_event_need_no_restriction(monkeypatch):
    # Only a restriction left with scalar inputs alone is recomputed per
    # event; with vectors everywhere, constant restrictions are read from the
    # pass (exactly 1.0 or 0.0, as the per-event path gives).
    tree = {"id": "T", "gate": "OR", "children": [
        {"id": "G", "gate": "AND", "children": [{"id": "B", "prob": [0.3, 0.4]},
                                                {"id": "C", "prob": [0.5, 0.6]}]},
        {"id": "A", "prob": [0.1, 0.2]}]}
    new, old = both_paths(monkeypatch, tree)
    compare(new, old, "vectors")
    calls = []
    monkeypatch.setattr(builder, "_ONE_PASS_IMPORTANCE", True)
    monkeypatch.setattr(builder.BDD, "let", lambda self, *a, **k: calls.append(a) or None)
    result = analyze(copy.deepcopy(tree))
    assert calls == []
    assert result["conditional_Q"]["A"]["true"] == 1.0 and type(result["conditional_Q"]["A"]["true"]) is float
