"""House events, INHIBIT gates and conditioning events (NUREG-0492 symbols)."""
import pytest

from faultree import analyze, minimal_cut_sets
from faultree.builder import normalize_tree


def _house_tree(state):
    # Train A fails, or train B fails while the cross-tie is in service.
    return {"id": "TOP", "gate": "OR", "children": [
        {"id": "A", "prob": 0.1},
        {"id": "G1", "gate": "AND", "children": [
            {"id": "B", "prob": 0.2},
            {"id": "XTIE", "event_type": "house", "state": state}]}]}


@pytest.mark.parametrize("state, expected", [(True, 1 - 0.9 * 0.8), (False, 0.1)])
def test_house_event_is_a_constant_switch(state, expected):
    result = analyze(_house_tree(state))
    assert result["Q"] == pytest.approx(expected, rel=1e-12)
    assert result["probabilities"]["XTIE"] == (1.0 if state else 0.0)
    # Not a variable: no probability needed, no importance, not in cut sets.
    assert "XTIE" not in result["importance"]
    assert "XTIE" not in result["assumptions"]["variable_order"]
    cuts = minimal_cut_sets(_house_tree(state))
    assert cuts["cut_sets"] == ([["A"], ["B"]] if state else [["A"]])
    assert cuts["complete"]


@pytest.mark.parametrize("state", [True, False])
def test_house_event_success_mode_is_the_complement(state):
    q = analyze(_house_tree(state))["Q"]
    tree = _house_tree(state)
    tree["children"][0]["prob"] = 0.9
    tree["children"][1]["children"][0]["prob"] = 0.8
    r = analyze(tree, success_mode=True)
    assert r["Q"] == pytest.approx(1 - q, rel=1e-12)


def test_house_event_in_expressions():
    result = analyze(_house_tree(False))
    assert result["expression"] == "OR(A, AND(B, FALSE))"
    assert result["symbolic"] == "(A + (B * 0))"


def test_house_event_can_be_shared():
    tree = {"id": "TOP", "gate": "AND", "children": [
        {"id": "G1", "gate": "OR", "children": [
            {"id": "A", "prob": 0.1}, {"id": "H", "event_type": "house", "state": False}]},
        {"id": "G2", "gate": "OR", "children": [
            {"id": "B", "prob": 0.2}, {"ref": "H"}]}]}
    assert analyze(tree)["Q"] == pytest.approx(0.02, rel=1e-12)


@pytest.mark.parametrize("bad, message", [
    ({"id": "H", "event_type": "house"}, "boolean 'state'"),
    ({"id": "H", "event_type": "house", "state": 1}, "boolean 'state'"),
    ({"id": "H", "event_type": "house", "state": True, "prob": 0.5}, "not 'prob'"),
])
def test_house_event_validation(bad, message):
    tree = {"id": "TOP", "gate": "OR", "children": [{"id": "A", "prob": 0.1}, bad]}
    with pytest.raises(ValueError, match=message):
        analyze(tree)


def _inhibit_tree():
    # Overpressure only causes rupture if the relief valve is also stuck.
    return {"id": "TOP", "gate": "OR", "children": [
        {"id": "LEAK", "prob": 1e-3},
        {"id": "RUPT", "gate": "INHIBIT", "children": [
            {"id": "OVERP", "gate": "OR", "children": [
                {"id": "CTRL", "prob": 0.01}, {"id": "OPER", "prob": 0.02}]},
            {"id": "RV_STUCK", "event_type": "conditioning", "prob": 0.05}]}]}


def test_inhibit_is_input_times_conditional_probability():
    result = analyze(_inhibit_tree())
    overp = 1 - 0.99 * 0.98
    rupt = overp * 0.05
    assert result["probabilities"]["RUPT"] == pytest.approx(rupt, rel=1e-12)
    assert result["Q"] == pytest.approx(1 - (1 - 1e-3) * (1 - rupt), rel=1e-12)
    # The conditioning event is a variable: it has importance and is in cut sets.
    assert "RV_STUCK" in result["importance"]
    cuts = minimal_cut_sets(_inhibit_tree())["cut_sets"]
    assert cuts == [["LEAK"], ["CTRL", "RV_STUCK"], ["OPER", "RV_STUCK"]]


def test_inhibit_success_mode_is_the_complement():
    q = analyze(_inhibit_tree())["Q"]
    tree = _inhibit_tree()
    probs = {"LEAK": 1 - 1e-3, "CTRL": 0.99, "OPER": 0.98, "RV_STUCK": 0.95}
    r = analyze(tree, probs, success_mode=True)
    assert r["Q"] == pytest.approx(1 - q, rel=1e-12)


@pytest.mark.parametrize("children, message", [
    ([{"id": "A", "prob": 0.1}], "exactly two inputs"),
    ([{"id": "A", "prob": 0.1}, {"id": "B", "prob": 0.2}], "exactly two inputs"),
    ([{"id": "C1", "event_type": "conditioning", "prob": 0.1},
      {"id": "C2", "event_type": "conditioning", "prob": 0.2}], "exactly two inputs"),
    ([{"id": "A", "prob": 0.1}, {"id": "B", "prob": 0.2},
      {"id": "C", "event_type": "conditioning", "prob": 0.1}], "exactly two inputs"),
])
def test_inhibit_validation(children, message):
    with pytest.raises(ValueError, match=message):
        analyze({"id": "TOP", "gate": "INHIBIT", "children": children})


def test_conditioning_event_outside_inhibit_is_rejected():
    tree = {"id": "TOP", "gate": "AND", "children": [
        {"id": "A", "prob": 0.1}, {"id": "C", "event_type": "conditioning", "prob": 0.5}]}
    with pytest.raises(ValueError, match="input of an INHIBIT gate"):
        analyze(tree)


def test_conditioning_event_cannot_be_shared():
    cond = {"id": "C", "event_type": "conditioning", "prob": 0.5}
    by_copy = {"id": "TOP", "gate": "OR", "children": [
        {"id": "I1", "gate": "INHIBIT", "children": [{"id": "A", "prob": 0.1}, cond]},
        {"id": "I2", "gate": "INHIBIT", "children": [{"id": "B", "prob": 0.1}, dict(cond)]}]}
    with pytest.raises(ValueError, match="more than once"):
        analyze(by_copy)
    by_ref = {"id": "TOP", "gate": "OR", "children": [
        {"id": "I1", "gate": "INHIBIT", "children": [{"id": "A", "prob": 0.1}, cond]},
        {"id": "I2", "gate": "INHIBIT", "children": [{"id": "B", "prob": 0.1}, {"ref": "C"}]}]}
    with pytest.raises(ValueError, match="cannot be shared"):
        analyze(by_ref)


def test_flat_format_carries_event_types_and_state():
    flat = {
        "ft_nodes": [
            {"label": "TOP", "gate": "OR", "branches": ["A", "RUPT", "G1"]},
            {"label": "RUPT", "gate": "INHIBIT", "branches": ["B", "C"]},
            {"label": "G1", "gate": "AND", "branches": ["D", "H"]}],
        "be_nodes": [
            {"label": "A", "prob": 0.1},
            {"label": "B", "prob": 0.2},
            {"label": "C", "event_type": "conditioning", "prob": 0.5},
            {"label": "D", "prob": 0.3},
            {"label": "H", "event_type": "house", "state": False}]}
    tree = normalize_tree(flat)
    house = tree["children"][2]["children"][1]
    assert house["event_type"] == "house" and house["state"] is False
    assert analyze(flat)["Q"] == pytest.approx(1 - 0.9 * (1 - 0.2 * 0.5), rel=1e-12)
    flat["be_nodes"][0]["event_type"] = "gate"
    with pytest.raises(ValueError, match="event_type"):
        normalize_tree(flat)
