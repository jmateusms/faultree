"""Reliability block diagrams: conversion to and from fault trees, analysis,
minimal path sets and the CLI options."""
import json
import math
import os
from itertools import combinations, product

import pytest

from faultree import (analyze, cli, minimal_cut_sets, minimal_path_sets,
                      rbd_to_tree, tree_to_rbd)
from faultree.builder import collect_basic_events, normalize_tree
from faultree.gui import engine
from faultree.rbd import is_rbd

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")
#: Example trees whose probabilities come from a file the tree does not name.
PROB_FILES = {"large_tree.json": "large_example_probs.csv"}


def load(name):
    with open(os.path.join(EXAMPLES, name), encoding="utf-8") as fh:
        return json.load(fh)


def example_files():
    return sorted(name for name in os.listdir(EXAMPLES) if name.endswith(".json"))


def prob_input(name, model):
    if name in PROB_FILES:
        return os.path.join(EXAMPLES, PROB_FILES[name])
    if model.get("prob_file"):
        return os.path.join(EXAMPLES, model["prob_file"])
    return None


# ------------------------------------------------------------------ oracle --
# Evaluates the diagram itself (paths of working blocks), sharing no code with
# the fault-tree conversion.
def _registry(item, out):
    if isinstance(item, dict) and "ref" not in item:
        if "id" in item:
            out[str(item["id"])] = item
        for child in item.get("blocks", []):
            _registry(child, out)
    return out


def _works(item, registry, up):
    if "ref" in item:
        return _works(registry[str(item["ref"])], registry, up)
    if "switch" in item:
        return item["switch"] == "closed"
    if "structure" in item:
        states = [_works(child, registry, up) for child in item["blocks"]]
        if item["structure"] == "series":
            return all(states)
        if item["structure"] == "parallel":
            return any(states)
        return sum(states) >= item["k"]
    return up[str(item["id"])]


def _blocks(rbd):
    return sorted(i for i, item in _registry(rbd, {}).items()
                  if "structure" not in item and "switch" not in item)


def oracle_q(rbd, probs=None):
    """System failure probability by enumerating every block state."""
    registry = _registry(rbd, {})
    blocks = _blocks(rbd)
    probs = probs or {b: registry[b]["prob"] for b in blocks}
    q = 0.0
    for states in product([True, False], repeat=len(blocks)):
        up = dict(zip(blocks, states))
        weight = 1.0
        for b in blocks:
            weight *= (1.0 - probs[b]) if up[b] else probs[b]
        if not _works(rbd, registry, up):
            q += weight
    return q


def oracle_path_sets(rbd):
    """Minimal sets of blocks whose working alone keeps the system working."""
    registry = _registry(rbd, {})
    blocks = _blocks(rbd)
    found = []
    for size in range(len(blocks) + 1):
        for subset in combinations(blocks, size):
            if any(set(p) <= set(subset) for p in found):
                continue
            if _works(rbd, registry, {b: b in subset for b in blocks}):
                found.append(subset)
    return sorted(map(list, found))


# ---------------------------------------------------------------- mapping --
def test_each_structure_becomes_its_dual_gate():
    rbd = {"model": "rbd", "id": "SYS", "structure": "series", "blocks": [
        {"id": "A", "prob": 0.1},
        {"structure": "parallel", "blocks": [{"id": "B", "prob": 0.2}, {"id": "C", "prob": 0.3}]},
        {"id": "V", "structure": "k_of_n", "k": 3, "blocks": [
            {"id": x, "prob": 0.1} for x in "DEFG"]},
        {"id": "SW", "switch": "open"},
        {"id": "U", "kind": "undeveloped", "prob": 0.01},
        {"ref": "A"}]}
    tree = rbd_to_tree(rbd)
    assert tree["gate"] == "OR" and tree["event_type"] == "top"
    a, par, vote, switch, undeveloped, ref = tree["children"]
    assert a == {"id": "A", "event_type": "basic", "gate": None, "prob": 0.1, "children": []}
    assert (par["id"], par["gate"], par["event_type"]) == ("SYS.2", "AND", "intermediate")
    # 3-out-of-4 working fails when at least 4 - 3 + 1 = 2 blocks fail.
    assert (vote["gate"], vote["k"]) == ("K_OF_N", 2)
    assert (switch["event_type"], switch["state"]) == ("house", True)
    assert undeveloped["event_type"] == "undeveloped"
    assert ref == {"ref": "A"}


def test_a_parallel_pair_with_a_condition_is_an_inhibit_gate():
    rbd = {"model": "rbd", "id": "RUPT", "structure": "parallel", "blocks": [
        {"id": "OVERP", "prob": 0.003},
        {"id": "RV", "kind": "conditioning", "prob": 0.05}]}
    tree = rbd_to_tree(rbd)
    assert tree["gate"] == "INHIBIT"
    assert tree["children"][1]["event_type"] == "conditioning"
    assert analyze(rbd)["Q"] == pytest.approx(0.003 * 0.05, rel=1e-12)
    assert tree_to_rbd(tree) == rbd


def test_xor_pand_and_k_zero_stay_fault_tree_blocks():
    for name in ("xor_simple.json", "priority_and.json"):
        rbd = tree_to_rbd(load(name))
        assert "fault_tree" in json.dumps(rbd)
    always = {"id": "T", "gate": "K_OF_N", "k": 0, "children": [{"id": "A", "prob": 0.5}]}
    assert tree_to_rbd(always) == {"model": "rbd", "fault_tree": always}
    assert analyze(tree_to_rbd(always))["Q"] == 1.0


def test_a_fault_tree_block_mixes_logic_into_a_diagram():
    xor = {"id": "X", "gate": "XOR", "children": [{"id": "B", "prob": 0.2}, {"id": "C", "prob": 0.3}]}
    rbd = {"model": "rbd", "id": "SYS", "structure": "series",
           "blocks": [{"id": "A", "prob": 0.1}, {"fault_tree": xor}]}
    exactly_one = 0.2 * 0.7 + 0.8 * 0.3
    assert analyze(rbd)["Q"] == pytest.approx(1 - 0.9 * (1 - exactly_one), rel=1e-12)


# --------------------------------------------------------------- analysis --
@pytest.mark.parametrize("name", ["rbd_cooling.json", "rbd_bridge.json"])
def test_examples_match_the_diagram_oracle(name):
    rbd = load(name)
    assert analyze(rbd)["Q"] == pytest.approx(oracle_q(rbd), rel=1e-12, abs=1e-15)


def test_open_switch_cuts_its_train():
    rbd = load("rbd_cooling.json")
    rbd["blocks"][1]["blocks"][1]["blocks"][0]["switch"] = "open"
    q = analyze(rbd)["Q"]
    assert q == pytest.approx(oracle_q(rbd), rel=1e-12)
    assert q > analyze(load("rbd_cooling.json"))["Q"]


def test_bridge_reliability_matches_the_closed_form():
    r = 0.9
    probs = {b: 1 - r for b in "ABCDE"}
    q = analyze(load("rbd_bridge.json"), probs)["Q"]
    assert 1 - q == pytest.approx(2 * r**2 + 2 * r**3 - 5 * r**4 + 2 * r**5, rel=1e-12)


def test_success_mode_reads_block_probabilities_as_reliabilities():
    rbd = load("rbd_cooling.json")
    probs = {b: 1 - q for b, q in
             ((b, _registry(rbd, {})[b]["prob"]) for b in _blocks(rbd))}
    r = analyze(rbd, probs, success_mode=True)["Q"]
    assert r == pytest.approx(1 - analyze(rbd)["Q"], rel=1e-12)


# ------------------------------------------------------------- round trips --
@pytest.mark.parametrize("name", [n for n in example_files() if is_rbd(load(n))])
def test_rbd_examples_survive_the_round_trip_unchanged(name):
    rbd = load(name)
    assert tree_to_rbd(rbd_to_tree(rbd)) == rbd


@pytest.mark.parametrize("name", [n for n in example_files() if not is_rbd(load(n))])
def test_fault_trees_survive_the_round_trip(name):
    tree = load(name)
    rbd = tree_to_rbd(tree)
    back = rbd_to_tree(rbd)
    # The diagram is a fixed point, and every event keeps its probability.
    assert tree_to_rbd(back) == rbd
    probs = prob_input(name, tree)
    before = analyze(tree, probs)["probabilities"]
    after = analyze(back, probs)["probabilities"]
    assert set(before) == set(after)
    for nid, value in before.items():
        assert after[nid] == pytest.approx(value, rel=1e-12, abs=1e-15), nid


# -------------------------------------------------------------- path sets --
@pytest.mark.parametrize("name", ["rbd_cooling.json", "rbd_bridge.json"])
def test_path_sets_match_the_diagram_oracle(name):
    rbd = load(name)
    found = minimal_path_sets(rbd, max_order=10)
    assert found["complete"]
    assert sorted(found["path_sets"]) == oracle_path_sets(rbd)


def test_bridge_path_and_cut_sets():
    rbd = load("rbd_bridge.json")
    assert sorted(minimal_path_sets(rbd)["path_sets"]) == [
        ["A", "B"], ["A", "D", "E"], ["B", "C", "E"], ["C", "D"]]
    assert sorted(minimal_cut_sets(rbd)["cut_sets"]) == [
        ["A", "C"], ["A", "D", "E"], ["B", "C", "E"], ["B", "D"]]


def test_path_sets_of_the_pressure_tank_match_brute_force():
    tree = normalize_tree(load("pressure_tank.json"))
    events = []
    collect_basic_events(tree, events)
    events = sorted(set(events))
    fails = lambda failed: analyze(tree, {e: float(e in failed) for e in events})["Q"] == 1.0
    expected = []
    for size in range(len(events) + 1):
        for subset in combinations(events, size):
            if any(set(p) <= set(subset) for p in expected):
                continue
            if not fails(set(events) - set(subset)):
                expected.append(subset)
    assert sorted(minimal_path_sets(tree)["path_sets"]) == sorted(map(list, expected))


@pytest.mark.parametrize("state, paths", [(True, [["A", "B"]]), (False, [["A"]])])
def test_path_sets_flip_house_events(state, paths):
    tree = {"id": "T", "gate": "OR", "children": [
        {"id": "A", "prob": 0.1},
        {"id": "G", "gate": "AND", "children": [
            {"id": "B", "prob": 0.2}, {"id": "H", "event_type": "house", "state": state}]}]}
    assert minimal_path_sets(tree)["path_sets"] == paths


def test_a_condition_that_does_not_hold_is_a_path():
    tree = {"id": "T", "gate": "INHIBIT", "children": [
        {"id": "A", "prob": 0.1}, {"id": "C", "event_type": "conditioning", "prob": 0.5}]}
    assert minimal_path_sets(tree)["path_sets"] == [["A"], ["C"]]


def test_path_sets_reject_xor():
    with pytest.raises(ValueError, match="monotone"):
        minimal_path_sets(load("xor_simple.json"))


def test_a_gate_that_always_fails_has_no_path_through_it():
    always = {"id": "K0", "gate": "K_OF_N", "k": 0, "children": [{"id": "A", "prob": 0.1}]}
    found = minimal_path_sets({"id": "T", "gate": "OR", "children": [always, {"id": "B", "prob": 0.1}]})
    assert found["path_sets"] == [] and found["complete"]
    found = minimal_path_sets({"id": "T", "gate": "AND", "children": [always, {"id": "B", "prob": 0.1}]})
    assert found["path_sets"] == [["B"]] and found["complete"]


def _random_rbd(rng, n_blocks):
    """A random diagram over blocks B0..B{n-1}: nested groups, with repeated
    blocks and switches."""
    names = [f"B{i}" for i in range(n_blocks)]
    defined = set()

    def item(depth):
        roll = rng.random()
        if depth >= 3 or roll < 0.45:
            if rng.random() < 0.08:
                return {"id": f"S{rng.randrange(10**6)}", "switch": rng.choice(["open", "closed"])}
            name = rng.choice(names)
            if name in defined:
                return {"ref": name}
            defined.add(name)
            return {"id": name, "prob": 0.1}
        size = rng.randint(1, 4)
        blocks = [item(depth + 1) for _ in range(size)]
        structure = rng.choice(["series", "parallel", "k_of_n"])
        group = {"id": f"G{rng.randrange(10**9)}", "structure": structure, "blocks": blocks}
        if structure == "k_of_n":
            group["k"] = rng.randint(1, size)
        return group

    top = {"model": "rbd", "id": "SYS", "structure": rng.choice(["series", "parallel"]),
           "blocks": [item(1) for _ in range(rng.randint(2, 4))]}
    return top


@pytest.mark.parametrize("seed", range(40))
def test_path_sets_of_random_diagrams_match_the_oracle(seed):
    import random
    rng = random.Random(seed)
    rbd = _random_rbd(rng, rng.randint(3, 9))
    found = minimal_path_sets(rbd, max_order=12)
    assert found["complete"]
    assert sorted(found["path_sets"]) == oracle_path_sets(rbd)


def test_bounded_path_sets_are_minimal_and_complete_to_their_order():
    import random
    nontrivial = 0
    for seed in range(60):
        for max_family, max_sets in [(3, 4), (4, 3)]:
            rbd = _random_rbd(random.Random(100 + seed), 9)
            oracle = oracle_path_sets(rbd)
            found = minimal_path_sets(rbd, max_order=12, max_family=max_family, max_sets=max_sets)
            listed = sorted(found["path_sets"])
            assert all(p in oracle for p in listed), seed
            if found["complete"]:
                assert listed == oracle, seed
                continue
            order = found["complete_to_order"]
            assert found["reason"] in ("max_family", "max_sets", "max_pairs"), seed
            assert listed == [p for p in oracle if len(p) <= order], seed
            assert any(len(p) > order for p in oracle), seed
            nontrivial += bool(listed)
    assert nontrivial >= 3  # truncated lists that still hold some paths


def test_a_long_series_diagram_has_one_long_path():
    rbd = {"model": "rbd", "id": "LINE", "structure": "series",
           "blocks": [{"id": f"B{i}", "prob": 0.01} for i in range(60)]}
    found = minimal_path_sets(rbd)
    assert found["path_sets"] == [] and not found["complete"]
    assert found["reason"] == "max_order" and found["complete_to_order"] == 10
    found = minimal_path_sets(rbd, max_order=60)
    assert found["complete"] and found["path_sets"] == [sorted(f"B{i}" for i in range(60))]


# -------------------------------------------------------------- validation --
def _rbd(*blocks, structure="series", **extra):
    return {"model": "rbd", "id": "SYS", "structure": structure, "blocks": list(blocks), **extra}


@pytest.mark.parametrize("rbd, message", [
    ({"model": "rbd", "structure": "series", "blocks": [{"id": "A"}]}, "top group needs an 'id'"),
    (_rbd({"id": "A"}, structure="bridge"), "structure 'bridge'"),
    (_rbd(), "non-empty 'blocks'"),
    (_rbd({"prob": 0.1}), "has no 'id'"),
    (_rbd({"id": "A", "kind": "house"}), "kind 'house'"),
    (_rbd({"id": "A"}, {"id": "B"}, structure="k_of_n", k=3), "1 <= k <= 2"),
    (_rbd({"id": "A"}, {"id": "B"}, structure="k_of_n", k=0), "1 <= k <= 2"),
    (_rbd({"id": "A"}, structure="parallel", k=1), "'k' applies to k_of_n"),
    (_rbd({"id": "A"}, {"id": "C", "kind": "conditioning"}), "parallel group of two"),
    (_rbd({"id": "A"}, {"id": "B"}, {"id": "C", "kind": "conditioning"}, structure="parallel"),
     "parallel group of two"),
    (_rbd({"id": "S", "switch": "half"}), "must be \"open\""),
    (_rbd({"id": "S", "switch": "open", "prob": 0.1}), "only its position"),
    (_rbd({"id": "G", "gate": "OR", "children": []}), "fault-tree fields"),
    (_rbd({"id": "G", "blocks": [{"id": "A"}]}), "'blocks' but no 'structure'"),
    (_rbd({"id": "G", "structure": "series", "switch": "open", "blocks": [{"id": "A"}]}), "mixes"),
    (_rbd({"fault_tree": {"id": "X", "gate": "OR", "children": [{"id": "A"}]}, "name": "x"}),
     "takes only 'fault_tree'"),
    (_rbd({"structure": "parallel", "blocks": [{"id": "A"}]}, {"id": "SYS.1"}), "generated"),
    ({"model": "rbd", "ref": "A"}, "cannot be a ref"),
])
def test_invalid_diagrams_are_rejected(rbd, message):
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
        normalize_tree(rbd)


def test_conflicting_repeated_ids_are_rejected_after_conversion():
    with pytest.raises(ValueError, match="Conflicting duplicate id 'A'"):
        normalize_tree(_rbd({"id": "A", "prob": 0.1}, {"id": "A", "prob": 0.2}))


# ---------------------------------------------------------- CLI, GUI, API --
def test_cli_converts_both_ways(capsys):
    cli.main([os.path.join(EXAMPLES, "rbd_bridge.json"), "--to-ft"])
    tree = json.loads(capsys.readouterr().out)
    assert tree["gate"] == "AND" and "model" not in tree
    cli.main([os.path.join(EXAMPLES, "pressure_tank.json"), "--to-rbd"])
    rbd = json.loads(capsys.readouterr().out)
    assert rbd["model"] == "rbd" and rbd["structure"] == "series"


def test_cli_lists_path_sets(capsys):
    cli.main(["--path-sets", os.path.join(EXAMPLES, "rbd_bridge.json")])
    assert json.loads(capsys.readouterr().out)["path_sets"][0] == ["A", "B"]
    cli.main([os.path.join(EXAMPLES, "rbd_bridge.json"), "--path-sets", "1"])
    found = json.loads(capsys.readouterr().out)
    assert found["path_sets"] == [] and not found["complete"]


@pytest.mark.parametrize("argv", [["--path-sets", "--reliability"], ["--path-sets", "2", "--cut-sets", "2"],
                                  ["--to-rbd", "--to-ft"]])
def test_cli_rejects_conflicting_options(argv, capsys):
    with pytest.raises(SystemExit):
        cli.main([os.path.join(EXAMPLES, "rbd_bridge.json"), *argv])


def test_cli_analyzes_a_diagram(capsys):
    cli.main([os.path.join(EXAMPLES, "rbd_cooling.json")])
    out = capsys.readouterr().out
    assert "Top event probability [COOLING]" in out


def test_gui_lists_and_loads_diagrams_as_trees():
    listing = {e["file"]: e for e in engine.list_examples()}
    assert listing["rbd_cooling.json"]["format"] == "rbd"
    loaded = engine.load_example("rbd_bridge.json")
    assert loaded["format"] == "rbd" and loaded["tree"]["gate"] == "AND"


def test_api_server_accepts_a_diagram():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from faultree.server import app
    response = TestClient(app).post("/analyze", json={"tree": load("rbd_bridge.json")})
    assert response.status_code == 200
    assert response.json()["top_event_probability"] == pytest.approx(
        oracle_q(load("rbd_bridge.json")), rel=1e-12)


# ------------------------------------------------------------- GUI Blocks --
def _rbd_signature(rbd):
    """Structure of an RBD as tree_to_rbd writes it, with repeated leaves as
    blocks and later occurrences of a group as repeats (as the editor holds
    them)."""
    registry, seen = _registry(rbd, {}), set()

    def sig(item):
        if "ref" in item:
            target = registry[str(item["ref"])]
            return ["block", item["ref"]] if "structure" not in target and "switch" not in target else ["ref", item["ref"]]
        if "fault_tree" in item:
            return ["module", item["fault_tree"]["id"]]
        if "switch" in item:
            return ["switch", item["id"], item["switch"]]
        if "structure" in item:
            if item["id"] in seen:
                return ["ref", item["id"]]
            seen.add(item["id"])
            return [item["structure"], item["id"], item.get("k"), [sig(b) for b in item["blocks"]]]
        return ["block", item["id"]]

    return sig(rbd)


def test_the_blocks_view_draws_what_tree_to_rbd_writes():
    from tests.test_gui import NODE, run_model_js
    if NODE is None:
        pytest.skip("Node.js is not installed")
    trees = {name: engine.load_example(name)["tree"] for name in example_files()}
    got = run_model_js(f"""
const trees = {json.dumps(trees)};
const out = {{}};
for (const [name, tree] of Object.entries(trees)) {{
  const {{ doc }} = M.importTree(tree);
  const sig = (it) => it.type === "clone" ? ["ref", it.n.ref] : it.type === "module" ? ["module", it.n.id]
    : it.type === "switch" ? ["switch", it.n.id, doc.events[it.n.id].state ? "open" : "closed"]
    : it.type === "block" ? ["block", it.n.id]
    : [{{ series: "series", parallel: "parallel", kofn: "k_of_n" }}[it.type], it.n.id, it.k ?? null, it.items.map(sig)];
  out[name] = sig(M.rbdOf(doc));
}}
console.log(JSON.stringify(out));""")
    for name, tree in trees.items():
        assert got[name] == _rbd_signature(tree_to_rbd(tree)), name


def test_point_analysis_lists_path_sets_with_their_reliability():
    tree = engine.load_example("rbd_cooling.json")["tree"]
    out = engine.analyze_point(tree)
    paths = out["path_sets"]
    assert paths["available"] and paths["complete"] and paths["complete_to_order"] is None
    assert sorted(row["events"] for row in paths["rows"]) == oracle_path_sets(load("rbd_cooling.json"))
    for row in paths["rows"]:
        assert row["p"] == pytest.approx(math.prod(1 - out["inputs"][e] for e in row["events"]))
    success = engine.analyze_point(tree, success_mode=True)["path_sets"]
    for row in success["rows"]:
        assert row["p"] == pytest.approx(math.prod(out["inputs"][e] for e in row["events"]))
    short = engine.analyze_point(tree, path_order=6)["path_sets"]
    assert short["rows"] == [] and short["reason"] == "max_order" and short["complete_to_order"] == 6
    assert not engine.analyze_point(load("xor_simple.json"))["path_sets"]["available"]
    with pytest.raises(ValueError, match="path_order"):
        engine.analyze_point(tree, path_order=0)


@pytest.mark.parametrize("name", example_files())
def test_saving_an_example_as_rbd_and_loading_it_back_keeps_the_model(name):
    from faultree.gui import server
    loaded = engine.load_example(name)
    rbd = server._post_routes()["/api/rbd"]({"tree": loaded["tree"]})["rbd"]
    assert is_rbd(rbd)
    back = engine.import_tree(rbd)
    assert back["format"] == "rbd"

    def model(tree):
        """With refs expanded (a repeat may come back as a ref or as a copy),
        without default fields (a leaf's basic type, null gate, no children)
        and a gate's event_type label ("top", "top event"...), not kept."""
        registry = {}

        def gather(node):
            if "ref" not in node and node.get("id") is not None:
                registry.setdefault(str(node["id"]), node)
            for child in node.get("children") or []:
                gather(child)

        def canon(node):
            if "ref" in node:
                node = registry[str(node["ref"])]
            if node.get("gate") and node.get("children"):
                out = {k: v for k, v in node.items() if k != "event_type"}
                out["children"] = [canon(child) for child in node["children"]]
                return out
            defaults = {"event_type": "basic", "gate": None, "children": []}
            return {k: v for k, v in node.items() if k not in defaults or v != defaults[k]}

        gather(tree)
        return canon(tree)

    assert model(back["tree"]) == model(normalize_tree(loaded["tree"]))


def test_rbd_export_rejects_bad_input():
    with pytest.raises(ValueError):
        engine.export_rbd(None)
    with pytest.raises(ValueError):
        engine.export_rbd({"id": "T", "gate": "NAND", "children": [{"id": "A"}]})


def test_block_diagram_edits_become_tree_edits():
    from tests.test_gui import NODE, run_model_js
    if NODE is None:
        pytest.skip("Node.js is not installed")
    tree = engine.load_example("rbd_cooling.json")["tree"]
    got = run_model_js(f"""
const {{ doc }} = M.importTree({json.dumps(tree)});
const find = (id) => {{ let r = null; M.walk(doc.root, (n, p, i) => {{ if (!r && (n.id === id)) r = {{ n, p, i }}; }}); return r; }};
const kids = (id) => find(id).n.children.map((c) => c.id || c.ref);
const out = {{}};
// in series inside a series group: a sibling right after the block
const e1 = M.addAround(doc, find("PUMP_A"), "series");
out.series = kids("TRAIN_A");
// in parallel inside a series group: the block and the new one in a new AND group
const e2 = M.addAround(doc, find("TANK"), "parallel");
const g = find("TANK").p;
out.parallel = [g.gate, g.children.map((c) => c.id), find(g.id).p.id];
// in parallel inside a parallel group: a sibling
M.addAround(doc, find("TANK"), "parallel");
out.parallel2 = kids(g.id).length;
// around the top: a new top group
const top = doc.root.id;
M.addAround(doc, {{ n: doc.root, p: null, i: 0 }}, "parallel");
out.root = [doc.root.gate, doc.root.children[0].id === top];
// k-out-of-n keeps the blocks that must work in the Blocks view, k in the tree view
const s = find("SENSORS").n;
const work = () => s.children.length - s.k + 1;
out.k0 = [s.k, work()];
M.addBlockTo(s, M.newEvent(doc), true); out.kAdd = [s.k, work(), s.children.length];
M.addBlockTo(s, M.newEvent(doc), false); out.kTree = [s.k, work(), s.children.length];
M.removeFrom(find("LS1"), true); out.kDel = [s.k, work(), s.children.length];
M.removeFrom(find("LS2"), false); out.kDelTree = [s.k, s.children.length];
// series -> parallel -> k-out-of-n -> series
const t = find("TRAIN_B").n, seen = [];
for (let i = 0; i < 3; i++) {{ M.cycleStructure(t); seen.push([t.gate, M.rbdStructure(t), t.children.length - t.k + 1]); }}
out.cycle = seen;
out.issues = M.validate(doc);
console.log(JSON.stringify(out));""")
    assert got["series"] == ["BUS_1", "PUMP_A", got["series"][2], "VALVE_A"]
    assert got["parallel"][0] == "AND" and got["parallel"][1][0] == "TANK" and got["parallel"][2] == "COOLING"
    assert got["parallel2"] == 3
    assert got["root"] == ["AND", True]
    assert got["k0"] == [2, 2]
    assert got["kAdd"] == [3, 2, 4]        # 2-out-of-4 must work: k = 3 failures
    assert got["kTree"] == [3, 3, 5]       # the tree view keeps k
    assert got["kDel"] == [2, 3, 4]
    assert got["kDelTree"] == [2, 3]
    assert [c[:2] for c in got["cycle"]] == [["AND", "parallel"], ["K_OF_N", "kofn"], ["OR", "series"]]
    assert got["cycle"][1][2] == 3  # four blocks: n - 1 = 3 must work
    assert got["issues"] == []


def test_new_events_take_a_reliability_in_success_mode():
    from tests.test_gui import NODE, run_model_js
    if NODE is None:
        pytest.skip("Node.js is not installed")
    got = run_model_js("""
const before = Object.values(M.emptyDoc().events).map((e) => e.prob);
M.setNewProb(0.99);
const after = Object.values(M.emptyDoc().events).map((e) => e.prob);
M.setNewProb(0.01);
console.log(JSON.stringify({ before, after }));""")
    assert got == {"before": [0.01, 0.01], "after": [0.99, 0.99]}


# ---------------------------------------------------------- review cases --
def _q(model):
    return analyze(model)["Q"]


@pytest.mark.parametrize("inner", ["XOR", "PAND", "K0"])
def test_identical_copies_across_a_fault_tree_block_round_trip(inner):
    a = {"id": "A", "prob": 0.1, "failure_model": {"dist": "exponential", "lambda": 1e-4}}
    b = {"id": "B", "prob": 0.2, "failure_model": {"dist": "exponential", "lambda": 2e-4}}
    gate = ({"id": "X", "gate": "K_OF_N", "k": 0, "children": [dict(a), dict(b)]} if inner == "K0"
            else {"id": "X", "gate": inner, "children": [dict(a), dict(b)]})
    tree = {"id": "T", "gate": "OR", "mission_time": 1000.0,
            "children": [{"id": "C", "prob": 0.05}, gate]}
    if inner != "PAND":  # a PAND shares no events: the copy goes outside it there
        tree["children"].insert(0, dict(a))
    rbd = tree_to_rbd(tree)
    assert _q(rbd) == pytest.approx(_q(tree), rel=1e-12)
    assert tree_to_rbd(rbd_to_tree(rbd)) == rbd


def test_a_flat_house_event_becomes_a_switch_that_loads():
    flat = {"ft_nodes": [{"label": "T", "gate": "AND", "branches": ["A", "H"]}],
            "be_nodes": [{"label": "A", "prob": 0.1}, {"label": "H", "event_type": "house", "state": True}]}
    rbd = tree_to_rbd(flat)
    assert _q(rbd) == pytest.approx(0.1)


def test_tree_to_rbd_validates_gates_and_names_unnamed_ones():
    with pytest.raises(ValueError, match="'k'"):
        tree_to_rbd({"id": "T", "gate": "K_OF_N", "children": [{"id": "A", "prob": 0.1}]})
    with pytest.raises(ValueError, match="k=3"):
        tree_to_rbd({"id": "T", "gate": "K_OF_N", "k": 3, "children": [{"id": "A", "prob": 0.1}, {"id": "B", "prob": 0.1}]})
    with pytest.raises(ValueError, match="top event needs an 'id'"):
        tree_to_rbd({"gate": "OR", "children": [{"id": "A", "prob": 0.1}]})
    tree = {"id": "T", "gate": "OR", "children": [
        {"gate": "AND", "children": [{"id": "A", "prob": 0.1}, {"id": "B", "prob": 0.2}]}, {"id": "C", "prob": 0.3}]}
    rbd = tree_to_rbd(tree)
    assert "id" not in rbd["blocks"][0] and _q(rbd) == pytest.approx(_q(tree))


@pytest.mark.parametrize("block, message", [
    ({"id": "H", "switch": "closed", "state": True}, "fault-tree fields"),
    ({"id": "A", "prob": 0.1, "event_type": "house"}, "fault-tree fields"),
    ({"ref": "A", "prob": 0.5}, "takes only 'ref'"),
    ({"fault_tree": {"id": "X", "gate": "XOR", "children": [
        {"id": "R", "structure": "series", "blocks": [{"id": "Q", "prob": 0.1}]}, {"id": "Z", "prob": 0.1}]}},
     "without RBD groups"),
])
def test_rbd_blocks_reject_fault_tree_fields_and_nesting(block, message):
    rbd = {"model": "rbd", "id": "S", "structure": "series", "blocks": [{"id": "A", "prob": 0.1}, block]}
    with pytest.raises(ValueError, match=message):
        rbd_to_tree(rbd)


def test_a_repeated_group_may_hold_unnamed_groups():
    inner = {"structure": "parallel", "blocks": [{"id": "A", "prob": 0.1}, {"id": "B", "prob": 0.2}]}
    group = {"id": "G", "structure": "series", "blocks": [{"id": "C", "prob": 0.05}, inner]}
    rbd = {"model": "rbd", "id": "S", "structure": "parallel", "blocks": [group, json.loads(json.dumps(group)), {"id": "D", "prob": 0.3}]}
    named = json.loads(json.dumps(rbd))
    named["blocks"][1] = {"ref": "G"}
    assert _q(rbd) == pytest.approx(_q(named))


def test_path_sets_report_the_pair_limit(monkeypatch):
    from faultree import cutsets
    ev = lambda eid: {"id": eid, "prob": 0.1}
    tree = {"id": "T", "gate": "OR", "children": [
        {"id": "G1", "gate": "AND", "children": [ev("A"), {"id": "O1", "gate": "OR", "children": [ev("X1"), ev("X2")]}]},
        {"id": "G2", "gate": "AND", "children": [{"ref": "A"}, {"id": "O2", "gate": "OR", "children": [ev("Y1"), ev("Y2")]}]}]}
    full = minimal_path_sets(tree)
    assert full["complete"] and sorted(full["path_sets"]) == [["A"], ["X1", "X2", "Y1", "Y2"]]
    monkeypatch.setattr(cutsets, "MAX_PAIRS", 3)
    bounded = minimal_path_sets(tree)
    assert bounded["reason"] == "max_pairs" and not bounded["complete"]
    assert bounded["complete_to_order"] == 1 and bounded["path_sets"] == [["A"]]


def test_path_sets_validate_the_tree_and_flag_every_drop():
    with pytest.raises(ValueError, match="no children"):
        minimal_path_sets({"id": "T", "gate": "OR", "children": []})
    with pytest.raises(ValueError, match="no 'id'"):
        minimal_path_sets({"id": "T", "gate": "OR", "children": [{"prob": 0.1}, {"id": "B", "prob": 0.1}]})
    one = minimal_path_sets({"id": "A", "prob": 0.1}, max_order=0)
    assert one["path_sets"] == [] and not one["complete"] and one["reason"] == "max_order"


def test_tree_keys_that_mean_something_in_an_rbd_are_not_copied():
    tree = {"id": "T", "gate": "OR", "model": "custom", "children": [
        {"id": "A", "prob": 0.1, "kind": "pump"}, {"id": "B", "prob": 0.2}]}
    rbd = tree_to_rbd(tree)
    assert rbd["model"] == "rbd" and "kind" not in rbd["blocks"][0]
    assert _q(rbd) == pytest.approx(1 - 0.9 * 0.8)


def test_a_complete_expansion_needs_no_bdd_check(monkeypatch):
    from faultree import cutsets
    monkeypatch.setattr(cutsets, "compile_tree", lambda *a, **k: (_ for _ in ()).throw(AssertionError("compiled")))
    assert minimal_path_sets(load("rbd_bridge.json"))["complete"]
