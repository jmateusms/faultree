"""PRIORITY AND gates quantified from failure-time models (faultree.dynamic)."""
import math

import numpy as np
import pytest

from faultree import analyze, minimal_cut_sets
from faultree.builder import compute_event_probabilities, build, normalize_tree
from faultree.dynamic import lifetime_cdf, ordered_cdf

T = 1000.0


def expo(eid, lam):
    return {"id": eid, "failure_model": {"dist": "exponential", "lambda": lam}}


def pand(gid, *children):
    return {"id": gid, "gate": "PAND", "children": list(children)}


def top(node, t=T):
    return {**node, "mission_time": t}


def two_exponentials(l1, l2, t=T):
    """P(T1 < T2 <= t) for independent exponential T1, T2."""
    return (1 - math.exp(-l2 * t)) - l2 / (l1 + l2) * (1 - math.exp(-(l1 + l2) * t))


@pytest.mark.parametrize("l1, l2", [(1e-3, 2e-3), (2e-3, 1e-3), (1e-5, 3e-4), (5e-3, 5e-3)])
def test_two_exponential_inputs_match_the_closed_form(l1, l2):
    result = analyze(top(pand("TOP", expo("A", l1), expo("B", l2))))
    assert result["Q"] == pytest.approx(two_exponentials(l1, l2), rel=1e-9)
    details = result["assumptions"]["pand"]["TOP"]
    assert details["steps"] == 4096 and details["mission_time"] == T
    assert details["abs_error_estimate"] < 1e-6


def test_order_matters_and_and_overstates_identical_inputs_by_two():
    lam = 1e-3
    f = 1 - math.exp(-lam * T)
    q = analyze(top(pand("TOP", expo("A", lam), expo("B", lam))))["Q"]
    assert q == pytest.approx(f * f / 2, rel=1e-9)
    ab = analyze(top(pand("TOP", expo("A", 1e-3), expo("B", 3e-3))))["Q"]
    ba = analyze(top(pand("TOP", expo("B", 3e-3), expo("A", 1e-3))))["Q"]
    f_a, f_b = 1 - math.exp(-1e-3 * T), 1 - math.exp(-3e-3 * T)
    assert ab + ba == pytest.approx(f_a * f_b, rel=1e-9)  # the two orders partition AND
    assert ab != pytest.approx(ba, rel=1e-3)


def test_three_identical_inputs_and_nesting():
    lam = 1e-3
    f = 1 - math.exp(-lam * T)
    flat = analyze(top(pand("TOP", expo("A", lam), expo("B", lam), expo("C", lam))))["Q"]
    assert flat == pytest.approx(f ** 3 / 6, rel=1e-9)
    nested = analyze(top(pand("TOP", pand("P1", expo("A", lam), expo("B", lam)), expo("C", lam))))
    assert nested["Q"] == pytest.approx(flat, rel=1e-9)
    assert nested["probabilities"]["P1"] == pytest.approx(f * f / 2, rel=1e-9)
    assert nested["expression"] == "PAND(PAND(A, B), C)"


def test_gate_inputs_use_the_subtree_failure_time():
    # OR(A, B) fails at min(TA, TB): exponential with the summed rate.
    or_input = top(pand("TOP", {"id": "G", "gate": "OR", "children": [expo("A", 1e-3), expo("B", 5e-4)]},
                        expo("C", 2e-3)))
    result = analyze(or_input)
    assert result["Q"] == pytest.approx(two_exponentials(1.5e-3, 2e-3), rel=1e-9)
    assert result["probabilities"]["G"] == pytest.approx(1 - math.exp(-1.5e-3 * T), rel=1e-12)
    assert result["probabilities"]["A"] == pytest.approx(1 - math.exp(-1e-3 * T), rel=1e-12)
    # AND(A, B) fails at max(TA, TB): integrate its CDF on a fine grid.
    and_input = top(pand("TOP", {"id": "G", "gate": "AND", "children": [expo("A", 1e-3), expo("B", 5e-4)]},
                         expo("C", 2e-3)))
    s = np.linspace(0, T, 400_001)
    mid = 0.5 * (s[1:] + s[:-1])
    g = (1 - np.exp(-1e-3 * mid)) * (1 - np.exp(-5e-4 * mid))
    reference = float(np.sum(g * np.diff(1 - np.exp(-2e-3 * s))))
    assert analyze(and_input)["Q"] == pytest.approx(reference, rel=1e-8)


def test_weibull_with_a_singular_density_at_zero():
    tree = top(pand("TOP", {"id": "A", "failure_model": {"dist": "weibull", "beta": 0.5, "eta": 4000.0}},
                    {"id": "B", "failure_model": {"dist": "weibull", "beta": 2.0, "eta": 1500.0}}))
    s = np.linspace(0, T, 2_000_001)
    mid = 0.5 * (s[1:] + s[:-1])
    reference = float(np.sum((1 - np.exp(-np.sqrt(mid / 4000.0))) * np.diff(1 - np.exp(-(s / 1500.0) ** 2))))
    result = analyze(tree)
    assert result["Q"] == pytest.approx(reference, rel=1e-8)
    assert result["assumptions"]["pand"]["TOP"]["abs_error_estimate"] < 1e-7


@pytest.mark.parametrize("model, expected", [
    ({"dist": "exponential", "lambda": 1e-3}, 1 - math.exp(-1.0)),
    ({"dist": "weibull", "beta": 2.0, "eta": 1000.0}, 1 - math.exp(-1.0)),
    ({"dist": "normal", "mu": 1000.0, "sigma": 100.0}, 0.5),
    ({"dist": "lognormal", "tmed": 1000.0, "s": 0.8}, 0.5),
    ({"dist": "gamma", "alpha": 2.0, "theta": 500.0}, 1 - 3 * math.exp(-2.0)),
    ({"dist": "uniform", "a": 0.0, "b": 4000.0}, 0.25),
    ({"dist": "poisson", "lambda": 2e-3, "k": 2}, 1 - 3 * math.exp(-2.0)),
])
def test_lifetime_cdfs_match_the_gui_formulas(model, expected):
    assert float(lifetime_cdf(model, np.array([T]))[0]) == pytest.approx(expected, rel=1e-12)


def test_ordered_cdf_of_one_input_is_its_cdf():
    f = np.linspace(0, 0.5, 11)
    assert np.allclose(ordered_cdf([f]), f)


def test_pand_in_a_static_tree_is_one_event():
    tree = top({"id": "TOP", "gate": "OR", "children": [
        pand("P", expo("A", 1e-3), expo("B", 2e-3)), {"id": "D", "prob": 0.01}]})
    p = two_exponentials(1e-3, 2e-3)
    result = analyze(tree)
    assert result["Q"] == pytest.approx(1 - (1 - p) * 0.99, rel=1e-9)
    assert set(result["importance"]) == {"P", "D"}
    assert result["expression"] == "OR(PAND(A, B), D)"
    assert minimal_cut_sets(tree)["cut_sets"] == [["D"], ["P"]]
    success = analyze(tree, {"D": 0.99}, success_mode=True)
    assert success["Q"] == pytest.approx(1 - result["Q"], rel=1e-12)
    assert success["probabilities"]["A"] == pytest.approx(math.exp(-1.0), rel=1e-12)
    assert success["expression"] == "AND(NOT(PAND(A, B)), D)"


def test_a_pand_gate_may_be_shared_by_ref():
    tree = top({"id": "TOP", "gate": "OR", "children": [
        {"id": "G1", "gate": "AND", "children": [pand("P", expo("A", 1e-3), expo("B", 2e-3)), {"id": "X", "prob": 0.5}]},
        {"id": "G2", "gate": "AND", "children": [{"ref": "P"}, {"id": "Y", "prob": 0.5}]}]})
    p = two_exponentials(1e-3, 2e-3)
    assert analyze(tree)["Q"] == pytest.approx(p * (1 - 0.25), rel=1e-9)


def test_house_events_inside_a_pand():
    first = top(pand("TOP", {"id": "H", "event_type": "house", "state": True}, expo("A", 1e-3)))
    assert analyze(first)["Q"] == pytest.approx(1 - math.exp(-1.0), rel=1e-9)
    second = top(pand("TOP", expo("A", 1e-3), {"id": "H", "event_type": "house", "state": True}))
    assert analyze(second)["Q"] == pytest.approx(0.0, abs=1e-15)
    switched_off = top(pand("TOP", {"id": "G", "gate": "OR", "children": [
        expo("A", 1e-3), {"id": "H", "event_type": "house", "state": False}]}, expo("B", 2e-3)))
    assert analyze(switched_off)["Q"] == pytest.approx(two_exponentials(1e-3, 2e-3), rel=1e-9)


def test_flat_format_carries_failure_models_and_mission_time():
    flat = {"mission_time": T, "time_unit": "h",
            "ft_nodes": [{"label": "TOP", "gate": "PAND", "branches": ["A", "B"]}],
            "be_nodes": [{"label": "A", "failure_model": {"dist": "exponential", "lambda": 1e-3}},
                         {"label": "B", "failure_model": {"dist": "exponential", "lambda": 2e-3}}]}
    assert normalize_tree(flat)["mission_time"] == T
    assert analyze(flat)["Q"] == pytest.approx(two_exponentials(1e-3, 2e-3), rel=1e-9)


def test_cli_event_probabilities_include_the_module():
    tree = top({"id": "TOP", "gate": "OR", "children": [
        pand("P", expo("A", 1e-3), expo("B", 2e-3)), {"id": "D", "prob": 0.01}]})
    bdd, _, _, _ = build(tree)
    probs = compute_event_probabilities(bdd, tree)
    assert probs["P"] == pytest.approx(two_exponentials(1e-3, 2e-3), rel=1e-9)
    assert probs["A"] == pytest.approx(1 - math.exp(-1.0), rel=1e-12)


def test_steps_parameter_and_error_estimate_shrink_together():
    tree = top(pand("TOP", expo("A", 1e-3), expo("B", 2e-3)))
    coarse = analyze(tree, pand_steps=64)["assumptions"]["pand"]["TOP"]["abs_error_estimate"]
    fine = analyze(tree, pand_steps=1024)["assumptions"]["pand"]["TOP"]["abs_error_estimate"]
    assert fine < coarse / 100  # second order: 16x the steps, ~256x smaller
    with pytest.raises(ValueError, match="even integer"):
        analyze(tree, pand_steps=101)


def _uniform_before(a1, b1, a2, b2):
    """P(T1 < T2) for T1 ~ U(a1, b1), T2 ~ U(a2, b2), by exact integration of
    the piecewise-linear F1 against the uniform density of T2."""
    def f1_integral(x):  # integral of F1 from -inf to x
        if x <= a1:
            return 0.0
        if x <= b1:
            return (x - a1) ** 2 / (2 * (b1 - a1))
        return (b1 - a1) / 2 + (x - b1)
    return (f1_integral(b2) - f1_integral(a2)) / (b2 - a2)


def _normal_before(mu1, mu2, sigma):
    return 0.5 * math.erfc(-((mu2 - mu1) / (sigma * math.sqrt(2))) / math.sqrt(2))


@pytest.mark.parametrize("m1, m2, t, exact", [
    # failure times far shorter than the mission time
    ({"dist": "exponential", "lambda": 5.0}, {"dist": "exponential", "lambda": 10.0}, 8760.0,
     two_exponentials(5.0, 10.0, 8760.0)),
    ({"dist": "exponential", "lambda": 20.0}, {"dist": "exponential", "lambda": 50.0}, 8760.0,
     two_exponentials(20.0, 50.0, 8760.0)),
    ({"dist": "exponential", "lambda": 0.5}, {"dist": "exponential", "lambda": 1.0}, 8760.0,
     two_exponentials(0.5, 1.0, 8760.0)),
    # narrow wear-out distributions
    ({"dist": "uniform", "a": 500.0, "b": 500.1}, {"dist": "uniform", "a": 500.05, "b": 500.2}, T,
     _uniform_before(500.0, 500.1, 500.05, 500.2)),
    ({"dist": "normal", "mu": 500.0, "sigma": 0.05}, {"dist": "normal", "mu": 500.06, "sigma": 0.05}, T,
     _normal_before(500.0, 500.06, 0.05)),
])
def test_steep_inputs_refine_the_grid(m1, m2, t, exact):
    tree = top(pand("TOP", {"id": "A", "failure_model": m1}, {"id": "B", "failure_model": m2}), t)
    result = analyze(tree)
    details = result["assumptions"]["pand"]["TOP"]
    assert result["Q"] == pytest.approx(exact, rel=2e-7)
    # The estimate bounds the error of the unextrapolated value, so it must
    # not be smaller than the error actually made.
    assert abs(result["Q"] - exact) <= max(3 * details["abs_error_estimate"], 1e-12)
    assert details["grid_points"] > 4097


def test_grid_refinement_has_a_memory_budget(monkeypatch):
    from faultree import dynamic
    monkeypatch.setattr(dynamic, "MAX_GRID_VALUES", 0)
    monkeypatch.setattr(dynamic, "REFINE_RISE", 2)
    tree = top(pand("TOP", expo("A", 5.0), expo("B", 10.0)), 8760.0)
    with pytest.raises(ValueError, match=r"PAND gate 'TOP'.*2 events x .*pand_steps=4096"):
        analyze(tree)


def test_inputs_failing_within_one_unresolvable_interval_are_an_error():
    tree = top(pand("TOP", expo("A", 1.0), expo("B", 2.0)), 1e25)
    with pytest.raises(ValueError, match="order of their failures"):
        analyze(tree)


def _weibull_first(beta, eta, lam, t):
    """P(T_A < T_B <= t), Weibull A, exponential B, by Gauss-Legendre in
    w = (u / eta)^beta, where the integrand is smooth."""
    x, wts = np.polynomial.legendre.leggauss(80)
    top_w = (t / eta) ** beta
    w = 0.5 * top_w * (x + 1)
    u = eta * w ** (1 / beta)
    integrand = -np.expm1(-w) * lam * np.exp(-lam * u) * (eta / beta) * w ** (1 / beta - 1)
    return float(0.5 * top_w * np.sum(wts * integrand))


@pytest.mark.parametrize("first, exact", [
    # an integrable singularity at 0: the CDF never resolves near 0
    ({"dist": "weibull", "beta": 0.1, "eta": 1000.0}, _weibull_first(0.1, 1000.0, 1e-3, T)),
    ({"dist": "weibull", "beta": 0.05, "eta": 1000.0}, _weibull_first(0.05, 1000.0, 1e-3, T)),
    # near-deterministic failure times, below the grid's float resolution
    ({"dist": "exponential", "lambda": 1e20}, 1 - math.exp(-1.0)),
    ({"dist": "normal", "mu": 500.0, "sigma": 1e-12}, math.exp(-0.5) - math.exp(-1.0)),
])
def test_unresolvable_intervals_that_cannot_matter_are_bounded(first, exact):
    tree = top(pand("TOP", {"id": "A", "failure_model": first}, expo("B", 1e-3)))
    result = analyze(tree)
    assert result["Q"] == pytest.approx(exact, rel=1e-9)
    assert result["assumptions"]["pand"]["TOP"]["abs_error_estimate"] < 1e-6


@pytest.mark.parametrize("model", [{"dist": "normal", "mu": 10.0, "sigma": 2.0},
                                   {"dist": "lognormal", "tmed": 10.0, "s": 1.0}])
def test_lifetime_cdf_takes_a_scalar(model):
    assert float(lifetime_cdf(model, 5.0)) == pytest.approx(float(lifetime_cdf(model, np.array([5.0]))[0]))


def test_a_gate_without_id_inside_a_pand():
    tree = top(pand("TOP", {"gate": "OR", "children": [expo("A", 1e-3), expo("B", 2e-3)]}, expo("C", 1e-3)))
    assert analyze(tree)["Q"] == pytest.approx(two_exponentials(3e-3, 1e-3), rel=1e-9)


def test_each_pand_module_gets_its_own_grid():
    def wearout(i, mu):
        return pand(f"P{i}", {"id": f"A{i}", "failure_model": {"dist": "normal", "mu": mu, "sigma": 20.0}},
                    {"id": f"B{i}", "failure_model": {"dist": "normal", "mu": mu + 10.0, "sigma": 20.0}})
    gates = [wearout(i, 500.0 + 300.0 * i) for i in range(20)]
    together = analyze(top({"id": "TOP", "gate": "OR", "children": gates}, 8760.0))
    alone = analyze(top(wearout(0, 500.0), 8760.0))["assumptions"]["pand"]["P0"]
    details = together["assumptions"]["pand"]
    assert details["P0"]["grid_points"] == alone["grid_points"] < 20_000
    assert details["P0"]["Q"] == alone["Q"]
    assert max(d["grid_points"] for d in details.values()) < 20_000


def test_tree_probabilities_inside_a_pand_are_left_out():
    tree = top({"id": "TOP", "gate": "OR", "children": [
        pand("P", {**expo("A", 1e-3), "prob": [0.9, 0.9, 0.9]}, expo("B", 2e-3)),
        {"id": "D", "prob": [0.01, 0.02]}]})
    result = analyze(tree)
    assert result["assumptions"]["sample_count"] == 2
    p = two_exponentials(1e-3, 2e-3)
    np.testing.assert_allclose(result["Q"], [1 - (1 - p) * 0.99, 1 - (1 - p) * 0.98], rtol=1e-9)


@pytest.mark.parametrize("back", [{"ref": "P"}, {"ref": "TOP"}])
def test_a_cycle_through_a_pand_is_reported(back):
    tree = top({"id": "TOP", "gate": "OR", "children": [
        pand("P", {"id": "H", "event_type": "house", "state": True},
             {"id": "G", "gate": "OR", "children": [expo("B", 1e-3), back]}),
        {"id": "X", "prob": 0.1}]})
    with pytest.raises(ValueError, match="Cycle detected.* -> P"):
        normalize_tree(tree)


def test_probabilities_for_events_inside_a_pand_are_rejected(tmp_path):
    tree = top({"id": "TOP", "gate": "OR", "children": [
        pand("P", expo("A", 1e-3), expo("B", 2e-3)), {"id": "D", "prob": 0.01}]})
    with pytest.raises(ValueError, match="inside a PAND gate: A"):
        analyze(tree, {"A": [0.9, 0.9, 0.9]})
    csv = tmp_path / "probs.csv"
    csv.write_text("B,D\n0.5,0.02\n")
    with pytest.raises(ValueError, match="inside a PAND gate: B"):
        analyze(tree, str(csv))
    # Forcing the gate itself is how a what-if sets it.
    assert analyze(tree, {"P": 1.0})["Q"] == 1.0


@pytest.mark.parametrize("tree, message", [
    (pand("TOP", expo("A", 1e-3), expo("B", 2e-3)), "mission_time"),
    (top(pand("TOP", expo("A", 1e-3))), "at least two inputs"),
    (top(pand("TOP", expo("A", 1e-3), {"id": "B", "prob": 0.1})), "failure_model"),
    (top(pand("TOP", expo("A", 1e-3), {"id": "B", "failure_model": {"dist": "binomial", "q": 0.1, "n": 5, "k": 1}})),
     "failure_model"),
    (top(pand("TOP", expo("A", 1e-3), {"id": "B", "failure_model": {"dist": "weibull", "beta": -1, "eta": 5}})),
     "invalid weibull"),
    (top(pand("TOP", {"id": "G", "gate": "XOR", "children": [expo("A", 1e-3), expo("B", 1e-3)]}, expo("C", 1e-3))),
     "keep a failure time"),
    (top(pand("TOP", {"id": "I", "gate": "INHIBIT", "children": [
        expo("A", 1e-3), {"id": "C", "event_type": "conditioning", "prob": 0.5}]}, expo("B", 1e-3))),
     r"\(INHIBIT\) cannot be inside"),
    (top(pand("TOP", expo("A", 1e-3), {"id": "G", "gate": "OR", "children": [
        expo("A", 1e-3), expo("B", 1e-3)]})), "share events"),
    (top({"id": "TOP", "gate": "OR", "children": [
        pand("P", expo("A", 1e-3), expo("B", 1e-3)), expo("A", 1e-3)]}), "also appear outside"),
    (top({"id": "TOP", "gate": "OR", "children": [
        pand("P", {"id": "G", "gate": "OR", "children": [expo("A", 1e-3), expo("B", 1e-3)]}, expo("C", 1e-3)),
        {"id": "H", "gate": "AND", "children": [{"ref": "G"}, {"id": "D", "prob": 0.1}]}]}), "also appear outside"),
])
def test_pand_validation(tree, message):
    with pytest.raises(ValueError, match=message):
        analyze(tree)
