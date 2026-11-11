"""Seeded random fault trees checked against full enumeration of their states:
top and intermediate probabilities, conditional Q, minimal cut and path sets
and the RBD round trip, in failure and success mode; PAND gates against a
fine-grid reference; every example file against the same enumeration.

The oracle shares no code with the engine: it evaluates gates on explicit
states and sums the probabilities of those states. Its sums have only
non-negative terms, as the engine's do, so results agree to 1e-12 relative
with no absolute floor."""
import copy
import itertools
import json
import math
import os
import random

import numpy as np
import pytest

from faultree import analyze, minimal_cut_sets, minimal_path_sets, rbd_to_tree, tree_to_rbd
from faultree.builder import _prepared_probabilities, gather_nodes, normalize_tree
from faultree.dynamic import lifetime_cdf, module_members

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")
REL = 1e-12
SEEDS = range(8)
TREES_PER_SEED = 40
#: Probabilities that exercise the edges: impossible, certain, rare.
EDGE_PROBS = (0.0, 1.0, 1e-9)


# --------------------------------------------------------------- generator --
def random_tree(rng, pool, depth=3):
    """A random tree over ``pool`` (events repeat), with every gate kind,
    K_OF_N from k = 0 to k = n, house events and INHIBIT gates."""
    counter = itertools.count()

    def node(level):
        if level == 0 or rng.random() < 0.3:
            if rng.random() < 0.08:
                return {"id": f"H{next(counter)}", "event_type": "house",
                        "state": rng.random() < 0.5}
            return {"id": rng.choice(pool)}
        gate = rng.choice(["AND", "OR", "XOR", "K_OF_N", "INHIBIT"])
        if gate == "INHIBIT":
            condition = {"id": f"C{next(counter)}", "event_type": "conditioning"}
            return {"id": f"G{next(counter)}", "gate": gate,
                    "children": [node(level - 1), condition]}
        n = rng.randint(1, 4)
        out = {"id": f"G{next(counter)}", "gate": gate,
               "children": [node(level - 1) for _ in range(n)]}
        if gate == "K_OF_N":
            out["k"] = rng.randint(0, n)
        return out

    tree = node(depth)
    while "children" not in tree:
        tree = node(depth)
    tree["id"] = "TOP"
    return tree


def random_probability(rng):
    return rng.choice(EDGE_PROBS + (round(rng.random(), 3), rng.random()))


def leaves(tree):
    out = []

    def visit(n):
        if "children" in n:
            for c in n["children"]:
                visit(c)
        elif n.get("event_type") != "house":
            out.append(n["id"])
    visit(tree)
    return sorted(set(out))


def with_probs(tree, probs):
    tree = copy.deepcopy(tree)

    def visit(n):
        if "children" in n:
            for c in n["children"]:
                visit(c)
        elif n.get("event_type") != "house":
            n["prob"] = probs[n["id"]]
    visit(tree)
    return tree


def random_case(rng):
    pool = [f"e{i}" for i in range(rng.randint(1, 6))]
    tree = random_tree(rng, pool)
    ids = leaves(tree)
    probs = {e: random_probability(rng) for e in ids}
    return with_probs(tree, probs), ids, probs


def random_cases():
    for seed in SEEDS:
        rng = random.Random(seed)
        for index in range(TREES_PER_SEED):
            yield f"{seed}-{index}", random_case(rng)


def monotone(node):
    return "children" not in node or (
        node["gate"] != "XOR" and all(monotone(c) for c in node["children"]))


# ------------------------------------------------------------------ oracle --
def fails(node, registry, state):
    """Failure of a fault-tree node in a state of its events (refs followed)."""
    if "ref" in node:
        return fails(registry[str(node["ref"])], registry, state)
    children = node.get("children") or []
    if not children and node.get("gate") != "PAND" or node.get("id") in state:
        if node.get("event_type") == "house":
            return node["state"]
        return state[node["id"]]
    values = [fails(c, registry, state) for c in children]
    gate = node["gate"]
    if gate in ("AND", "INHIBIT"):
        return all(values)
    if gate == "OR":
        return any(values)
    if gate == "XOR":
        return sum(values) == 1
    if gate == "K_OF_N":
        return sum(values) >= node["k"]
    raise AssertionError(f"oracle has no gate {gate}")


def enumerate_states(ids, inputs, success_mode):
    """(failure state, weight, weight without each event) for every state.
    Engine inputs are reliabilities in success mode: the weight of a working
    event is its input itself, never 1 - (1 - r)."""
    for values in itertools.product((False, True), repeat=len(ids)):
        factors = []
        for failed, event in zip(values, ids):
            p = inputs[event]
            factors.append((1.0 - p if failed else p) if success_mode else (p if failed else 1.0 - p))
        without = {}
        for skip, event in enumerate(ids):
            w = 1.0
            for j, f in enumerate(factors):
                if j != skip:
                    w *= f
            without[event] = w
        yield dict(zip(ids, values)), math.prod(factors), without


def oracle(tree, ids, inputs, success_mode, nodes=None):
    """Top, per-node and conditional probabilities by enumeration. ``inputs``
    are engine inputs (reliabilities in success mode); ``nodes`` maps result
    ids to the nodes whose probability is wanted."""
    registry = {}
    gather_nodes(tree, registry)
    if nodes is None:
        nodes = {nid: n for nid, n in registry.items() if n.get("children")}
    # In success mode the engine reports the probability that a node works,
    # and forcing an event "true" means forcing it to work.
    counted = (lambda failed: not failed) if success_mode else (lambda failed: failed)
    top = 0.0
    probs = dict.fromkeys(nodes, 0.0)
    cond = {e: {"true": 0.0, "false": 0.0} for e in ids}
    for state, weight, without in enumerate_states(ids, inputs, success_mode):
        top_hit = counted(fails(tree, registry, state))
        if top_hit:
            top += weight
        for nid, n in nodes.items():
            if counted(fails(n, registry, state)):
                probs[nid] += weight
        if top_hit:
            for e in ids:
                forced = "true" if counted(state[e]) else "false"
                cond[e][forced] += without[e]
    return top, probs, cond


def close(actual, expected):
    return math.isclose(actual, expected, rel_tol=REL, abs_tol=0.0)


def assert_matches(result, top, probs, cond, where):
    assert close(result["Q"], top), (where, "Q", result["Q"], top)
    for nid, expected in probs.items():
        got = result["probabilities"][nid]
        assert close(got, expected), (where, nid, got, expected)
    for e, expected in cond.items():
        got = result["conditional_Q"][e]
        assert close(got["true"], expected["true"]), (where, e, "true", got, expected)
        assert close(got["false"], expected["false"]), (where, e, "false", got, expected)
        assert result["birnbaum"][e] == got["true"] - got["false"]


def minimal_sets(ids, hits):
    """Minimal subsets s of ids with hits(s), shortest first."""
    found = []
    for size in range(len(ids) + 1):
        for candidate in itertools.combinations(ids, size):
            if hits(set(candidate)) and not any(set(f) <= set(candidate) for f in found):
                found.append(candidate)
    return {frozenset(f) for f in found}


# ------------------------------------------------------------ random trees --
CASES = list(random_cases())


@pytest.mark.parametrize("seed", SEEDS)
def test_random_trees_match_enumeration(seed):
    for name, (tree, ids, probs) in CASES:
        if not name.startswith(f"{seed}-"):
            continue
        for success_mode in (False, True):
            result = analyze(copy.deepcopy(tree), success_mode=success_mode)
            assert_matches(result, *oracle(tree, ids, probs, success_mode),
                           (name, success_mode, tree))
            if not success_mode:
                importance = result["importance"]
                assert set(importance) == set(ids)
            else:
                assert result["importance"] is None


@pytest.mark.parametrize("seed", SEEDS)
def test_random_cut_and_path_sets_match_enumeration(seed):
    checked = 0
    for name, (tree, ids, _) in CASES:
        if not name.startswith(f"{seed}-") or not monotone(tree):
            continue
        registry = {}
        gather_nodes(tree, registry)

        def top_fails(failed):
            return fails(tree, registry, {e: e in failed for e in ids})

        cuts = minimal_cut_sets(copy.deepcopy(tree), max_order=len(ids))
        assert cuts["complete"], (name, tree)
        assert {frozenset(c) for c in cuts["cut_sets"]} == minimal_sets(ids, top_fails), (name, tree)
        paths = minimal_path_sets(copy.deepcopy(tree))
        assert paths["complete"], (name, tree)
        expected = minimal_sets(ids, lambda working: not top_fails(set(ids) - working))
        assert {frozenset(p) for p in paths["path_sets"]} == expected, (name, tree)
        checked += 1
    assert checked >= 10


@pytest.mark.parametrize("seed", SEEDS)
def test_random_rbd_round_trip(seed):
    for name, (tree, ids, _) in CASES:
        if not name.startswith(f"{seed}-"):
            continue
        rbd = tree_to_rbd(copy.deepcopy(tree))
        back = rbd_to_tree(copy.deepcopy(rbd))
        for success_mode in (False, True):
            expected = analyze(copy.deepcopy(tree), success_mode=success_mode)
            for model in (rbd, back):
                got = analyze(copy.deepcopy(model), success_mode=success_mode)
                assert close(got["Q"], expected["Q"]), (name, success_mode, got["Q"], expected["Q"])
                for e in ids:
                    assert got["conditional_Q"][e] == pytest.approx(
                        expected["conditional_Q"][e], rel=REL, abs=0.0), (name, e)


def test_generator_reaches_the_edge_cases():
    seen = set()
    for _, (tree, ids, probs) in CASES:
        def visit(n, parent_ids):
            if "children" in n:
                k, size = n.get("k"), len(n["children"])
                seen.add(n["gate"])
                if k == 0:
                    seen.add("k=0")
                if k == size and size > 1:
                    seen.add("k=n")
                for c in n["children"]:
                    visit(c, parent_ids)
            else:
                seen.add(n.get("event_type", "basic"))
                if n.get("event_type") not in ("house", "conditioning"):
                    if n["id"] in parent_ids:
                        seen.add("repeated")
                    parent_ids.add(n["id"])
        visit(tree, set())
        seen.update(f"p={p}" for p in probs.values() if p in EDGE_PROBS)
    assert {"AND", "OR", "XOR", "K_OF_N", "INHIBIT", "k=0", "k=n", "house",
            "conditioning", "repeated", "p=0.0", "p=1.0", "p=1e-09"} <= seen


# -------------------------------------------------------------------- PAND --
def stieltjes_reference(models, mission_time, points=400_001):
    """P(T_1 < ... < T_n <= t) on a fine uniform grid: G_1 = F_1 and
    G_k(s) = integral of G_{k-1} dF_k, by the midpoint of G on each step."""
    s = np.linspace(0.0, mission_time, points)
    g = lifetime_cdf(models[0], s)
    for model in models[1:]:
        dF = np.diff(lifetime_cdf(model, s))
        g = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * dF)])
    return float(g[-1])


PAND_CASES = [
    [{"dist": "exponential", "lambda": 2e-5}, {"dist": "weibull", "beta": 1.5, "eta": 30000.0}],
    [{"dist": "weibull", "beta": 0.5, "eta": 4000.0}, {"dist": "exponential", "lambda": 1e-3}],
    [{"dist": "lognormal", "tmed": 3000.0, "s": 0.8}, {"dist": "normal", "mu": 2000.0, "sigma": 600.0}],
    [{"dist": "uniform", "a": 500.0, "b": 6000.0}, {"dist": "weibull", "beta": 3.0, "eta": 5000.0},
     {"dist": "exponential", "lambda": 1e-4}],
    [{"dist": "normal", "mu": 4000.0, "sigma": 1500.0}, {"dist": "uniform", "a": 1000.0, "b": 9000.0}],
]


@pytest.mark.parametrize("models", PAND_CASES)
def test_pand_matches_a_fine_grid_reference(models):
    mission_time = 8760.0
    tree = {"id": "T", "gate": "PAND", "mission_time": mission_time,
            "children": [{"id": f"x{j}", "failure_model": m} for j, m in enumerate(models)]}
    result = analyze(tree)
    expected = stieltjes_reference(models, mission_time)
    estimate = result["assumptions"]["pand"]["T"]["abs_error_estimate"]
    # The reference's own error (its change when the grid is 4x finer) is
    # about 1e-8 for the Weibull with beta 0.5 and below 2e-12 for the others,
    # well under the engine's estimates.
    assert abs(result["Q"] - expected) <= max(estimate, 1e-9), (result["Q"], expected, estimate)


# ---------------------------------------------------------------- examples --
#: Every example file and the modes the enumeration checks it in.
ORACLE_EXAMPLES = {
    "basic_tree.json": (False, True),
    "connector.json": (False, True),
    "cooling_system.json": (False, True),
    "flat_tree.json": (False, True),
    "flat_tree_success.json": (False, True),
    "fta3.json": (False, True),
    "fta3_success.json": (False, True),
    "fta4.json": (False, True),
    "fta4_success.json": (False, True),
    "fta4b.json": (False, True),
    "kn_big.json": (False, True),
    "kn_simple.json": (False, True),
    "large_tree.json": (False, True),
    "nureg_symbols.json": (False, True),
    "pressure_tank.json": (False, True),
    "priority_and.json": (False, True),
    "rbd_bridge.json": (False, True),
    "rbd_cooling.json": (False, True),
    "ref_example.json": (False, True),
    "xor_big.json": (False, True),
    "xor_kn_big.json": (False, True),
    "xor_kn_big_linear.json": (False, True),
    "xor_simple.json": (False, True),
}
#: Example trees whose probabilities come from a file the tree does not name.
PROB_FILES = {"large_tree.json": "large_example_probs.csv"}


def test_every_example_file_is_in_the_oracle_list():
    files = {name for name in os.listdir(EXAMPLES) if name.endswith(".json")}
    assert files == set(ORACLE_EXAMPLES), (
        "add new example files to ORACLE_EXAMPLES: " + ", ".join(sorted(files ^ set(ORACLE_EXAMPLES))))


def rbd_fails(item, registry, state):
    """Failure of an RBD item, from the diagram itself: a group fails when
    it has no path of working blocks."""
    if "ref" in item:
        return rbd_fails(registry[str(item["ref"])], registry, state)
    if "switch" in item:
        return item["switch"] == "open"
    if "fault_tree" in item:
        return fails(item["fault_tree"], {}, state)
    if "structure" in item:
        failed = [rbd_fails(b, registry, state) for b in item["blocks"]]
        working = len(failed) - sum(failed)
        if item["structure"] == "series":
            return working < len(failed)
        if item["structure"] == "parallel":
            return working == 0
        return working < item["k"]
    return state[item["id"]]


def rbd_registry(item, out):
    if isinstance(item, dict) and "ref" not in item:
        if "id" in item:
            out[str(item["id"])] = item
        for block in item.get("blocks", []):
            rbd_registry(block, out)
    return out


def rbd_oracle(rbd, ids, inputs, success_mode):
    registry = rbd_registry(rbd, {})
    groups = {nid: item for nid, item in registry.items() if "structure" in item}
    counted = (lambda f: not f) if success_mode else (lambda f: f)
    top, probs = 0.0, dict.fromkeys(groups, 0.0)
    cond = {e: {"true": 0.0, "false": 0.0} for e in ids}
    for state, weight, without in enumerate_states(ids, inputs, success_mode):
        hit = counted(rbd_fails(rbd, registry, state))
        top += weight if hit else 0.0
        for nid, item in groups.items():
            if counted(rbd_fails(item, registry, state)):
                probs[nid] += weight
        if hit:
            for e in ids:
                cond[e]["true" if counted(state[e]) else "false"] += without[e]
    return top, probs, cond


def example_case(name, success_mode):
    with open(os.path.join(EXAMPLES, name), encoding="utf-8") as fh:
        model = json.load(fh)
    prob_file = PROB_FILES.get(name) or model.get("prob_file")
    prob_input = os.path.join(EXAMPLES, prob_file) if prob_file else None
    result = analyze(copy.deepcopy(model), prob_input, success_mode=success_mode)
    tree = normalize_tree(copy.deepcopy(model))
    # Input resolution (files, PAND module values) is the engine's; the
    # oracle checks the logic built on those inputs.
    inputs = _prepared_probabilities(tree, prob_input, shuffle=False, seed=0, allow_missing=False,
                                     resample_independent=False, success_mode=success_mode)
    return model, tree, inputs, result


def _at(value, index):
    """Sample ``index`` of a result that is a scalar when constant."""
    flat = np.asarray(value, dtype=float).reshape(-1)
    return float(flat[index] if flat.size > 1 else flat[0])


def _sample(inputs, index):
    return {k: _at(v, index) for k, v in inputs.items()}


@pytest.mark.parametrize("name,success_mode",
                         [(n, m) for n, modes in ORACLE_EXAMPLES.items() for m in modes])
def test_example_matches_enumeration(name, success_mode):
    model, tree, inputs, result = example_case(name, success_mode)
    ids = sorted(result["conditional_Q"])
    registry = {}
    gather_nodes(tree, registry)
    inside = module_members(tree, registry)
    nodes = {nid: n for nid, n in registry.items()
             if n.get("children") and n.get("gate") != "PAND" and nid not in inside}
    count = max(np.asarray(v).size for v in inputs.values())
    for index in range(count):
        scalar = _sample(inputs, index)
        if model.get("model") == "rbd":
            expected = rbd_oracle(model, ids, scalar, success_mode)
        else:
            expected = oracle(tree, ids, scalar, success_mode, nodes)
        picked = {
            "Q": _at(result["Q"], index),
            "probabilities": {nid: _at(v, index) for nid, v in result["probabilities"].items()},
            "conditional_Q": {e: {k: _at(v, index) for k, v in c.items()}
                              for e, c in result["conditional_Q"].items()},
            "birnbaum": {e: _at(v, index) for e, v in result["birnbaum"].items()},
        }
        assert_matches(picked, *expected, (name, success_mode, index))
