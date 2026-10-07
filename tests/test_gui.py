"""GUI backend: engine helpers, the HTTP server and the CLI entry point."""
import base64
import json
import math
import os
import re
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import zlib

import numpy as np
import pytest

from faultree import analyze, minimal_cut_sets
from faultree.builder import to_jsonable
from faultree.gui import engine
from faultree.gui.engine import GuiError
from faultree.gui.server import STATIC_DIR, make_server

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    with open(os.path.join(REPO, "examples", name), encoding="utf-8") as fh:
        return json.load(fh)


def leaf(event_id, prob):
    return {"id": event_id, "prob": prob}


def strict_json(obj):
    return json.loads(json.dumps(obj, allow_nan=False))


# --------------------------------------------------------------- examples --
def test_pressure_tank_matches_the_handbook_cut_sets():
    tree = load("pressure_tank.json")
    found = minimal_cut_sets(tree)
    assert found["complete"]
    assert sorted(map(tuple, found["cut_sets"])) == sorted(
        [("T",), ("K2",), ("S", "S1"), ("K1", "S"), ("R", "S")])


def test_examples_are_listed_and_load_with_their_probability_files():
    listing = {e["file"]: e for e in engine.list_examples()}
    assert {"pressure_tank.json", "fta4b.json", "flat_tree.json"} <= set(listing)
    tank = listing["pressure_tank.json"]
    assert (tank["basic_events"], tank["gates"]) == (6, 5)
    assert tank["title_pt"] == "Ruptura do tanque de pressão"
    assert listing["flat_tree.json"]["format"] == "flat"
    assert listing["fta3_success.json"]["success_mode"] is True

    fta4b = engine.load_example("fta4b.json")
    assert fta4b["prob_file"] == "fta4b_probs.csv"
    assert len(fta4b["probs"]["1"]) == 15
    large = engine.load_example("large_tree.json")
    assert set(large["probs"]) == {"E1", "E2", "E3"}
    flat = engine.load_example("flat_tree.json")
    assert flat["format"] == "flat" and flat["tree"]["id"] == "LoIRC2tE_FT"


@pytest.mark.parametrize("name", ["../README.md", "../pyproject.toml", "nope.json", "a/b.json", 3])
def test_unknown_examples_are_rejected(name):
    with pytest.raises(GuiError):
        engine.load_example(name)


# ------------------------------------------------------------------ import --
def test_import_accepts_recursive_flat_and_request_bodies():
    recursive = load("basic_tree.json")
    assert engine.import_tree(recursive) == {"tree": recursive, "format": "recursive"}
    flat = engine.import_tree(load("flat_tree.json"))
    assert flat["format"] == "flat" and flat["tree"]["gate"] == "OR"
    body = engine.import_tree({"tree": recursive, "probs": {"BE1": 0.5}})
    assert body["probs"] == {"BE1": 0.5}
    named = engine.import_tree(load("fta4b.json"))
    assert named["prob_file_missing"] == "fta4b_probs.csv"
    with pytest.raises(ValueError):
        engine.import_tree({"id": "T", "gate": "AND", "children": "x"})
    with pytest.raises(GuiError):
        engine.import_tree([1, 2])


# ----------------------------------------------------------- point analysis --
def test_point_analysis_matches_analyze_and_is_strict_json():
    tree = load("pressure_tank.json")
    out = strict_json(engine.analyze_point(tree))
    ref = analyze(tree)
    assert out["Q"] == pytest.approx(ref["Q"], rel=1e-15)
    for key, value in ref["probabilities"].items():
        assert out["probabilities"][key] == pytest.approx(value, rel=1e-15)
    assert out["importance"]["S"]["raw"] == pytest.approx(ref["importance"]["S"]["raw"])
    assert out["basic_events"] == ["T", "K2", "S", "S1", "K1", "R"]
    assert out["inputs"]["S"] == 0.01


def test_cut_set_table_has_probabilities_shares_and_approximations():
    out = engine.analyze_point(load("pressure_tank.json"))
    cs = out["cut_sets"]
    assert cs["available"] and cs["complete"] and cs["reason"] is None
    assert [row["events"] for row in cs["rows"]][:2] == [["K2"], ["S", "S1"]]
    for row in cs["rows"]:
        expected = math.prod(out["inputs"][e] for e in row["events"])
        assert row["p"] == pytest.approx(expected)
        assert row["share"] == pytest.approx(expected / out["Q"])
        assert row["order"] == len(row["events"])
    ps = [row["p"] for row in cs["rows"]]
    assert cs["rare_event"] == pytest.approx(sum(ps))
    assert cs["mcub"] == pytest.approx(1 - math.prod(1 - p for p in ps))
    assert cs["rare_event"] >= cs["mcub"] >= out["Q"]
    assert cs["by_order"] == {"1": 2, "2": 3}


def test_truncated_cut_sets_are_flagged():
    cs = engine.analyze_point(load("pressure_tank.json"), max_order=1)["cut_sets"]
    assert cs["available"] and not cs["complete"] and cs["truncated"]
    assert cs["reason"] == "max_order"
    assert sorted(r["events"][0] for r in cs["rows"]) == ["K2", "T"]


def test_xor_tree_reports_why_cut_sets_are_unavailable():
    out = engine.analyze_point(load("xor_simple.json"))
    assert out["cut_sets"]["available"] is False
    assert "XOR" in out["cut_sets"]["error"]
    assert out["Q"] == pytest.approx(analyze(load("xor_simple.json"))["Q"])


def test_success_mode_reports_reliability_without_cut_sets_or_ratios():
    tree = load("fta3_success.json")
    tree = json.loads(json.dumps(tree))
    for child in tree["children"]:  # scalar reliabilities for the point analysis
        for node in [child] + child.get("children", []):
            if isinstance(node.get("prob"), list):
                node["prob"] = node["prob"][0]
    out = engine.analyze_point(tree, success_mode=True)
    assert out["mode"] == "success"
    assert out["Q"] == pytest.approx(analyze(tree, success_mode=True)["Q"])
    assert out["importance"] is None
    assert out["cut_sets"] == {"available": False, "error": "success_mode"}


def test_point_analysis_rejects_sample_vectors_and_bad_options():
    with pytest.raises(GuiError, match="uncertainty"):
        engine.analyze_point(load("fta4.json"))
    with pytest.raises(GuiError):
        engine.analyze_point(load("basic_tree.json"), max_order=0)
    with pytest.raises(GuiError):
        engine.analyze_point(load("basic_tree.json"), success_mode="yes")


def test_bdd_drawing_is_a_plain_bdd_whose_root_is_exact_q():
    tree = {"id": "TOP", "gate": "XOR", "children": [leaf("A", 0.1), leaf("B", 0.2), leaf("C", 0.3)]}
    out = engine.analyze_point(tree)
    graph = out["bdd"]["graph"]
    assert graph["p"] == pytest.approx(out["Q"])
    nodes = {n["id"]: n for n in graph["nodes"]}
    for node in graph["nodes"]:
        for kid in (node["high"], node["low"]):
            assert kid in ("0", "1") or kid in nodes
        value = lambda k: 1.0 if k == "1" else 0.0 if k == "0" else nodes[k]["p"]
        assert node["p"] == pytest.approx(node["q"] * value(node["high"]) + (1 - node["q"]) * value(node["low"]))
    # brute force: P(exactly one of A, B, C)
    a, b, c = 0.1, 0.2, 0.3
    exact = a * (1 - b) * (1 - c) + (1 - a) * b * (1 - c) + (1 - a) * (1 - b) * c
    assert graph["p"] == pytest.approx(exact)
    assert engine.analyze_point(tree, bdd_limit=1)["bdd"]["graph"] is None


# ----------------------------------------------------------------- what-if --
def test_what_if_equals_the_conditional_probabilities_of_analyze():
    tree = load("pressure_tank.json")
    ref = analyze(tree)
    failed = engine.what_if(tree, {"S": "failed"})
    assert failed["Q"] == pytest.approx(ref["conditional_Q"]["S"]["true"])
    assert failed["base_Q"] == pytest.approx(ref["Q"])
    working = engine.what_if(tree, {"S": "working"})
    assert working["Q"] == pytest.approx(ref["conditional_Q"]["S"]["false"])
    both = engine.what_if(tree, {"S": "failed", "K1": "failed"})
    assert both["Q"] == pytest.approx(1.0)
    assert both["probabilities"]["E5"] == pytest.approx(1.0)


def test_what_if_in_success_mode_sets_failed_events_to_reliability_zero():
    tree = {"id": "TOP", "gate": "AND", "children": [leaf("A", 0.9), leaf("B", 0.8)]}
    ref = analyze(tree, success_mode=True)
    out = engine.what_if(tree, {"A": "failed"}, success_mode=True)
    assert out["Q"] == pytest.approx(ref["conditional_Q"]["A"]["false"])
    assert out["Q"] == pytest.approx(0.8)  # parallel pair: B alone keeps it working


@pytest.mark.parametrize("forced", [{"NOPE": "failed"}, {"S": "broken"}, ["S"]])
def test_what_if_rejects_bad_input(forced):
    with pytest.raises(GuiError):
        engine.what_if(load("pressure_tank.json"), forced)


# ------------------------------------------------------------- uncertainty --
def test_lognormal_median_and_error_factor_are_honoured():
    rng = np.random.default_rng(1)
    values, clipped = engine.sample_distribution({"dist": "lognormal", "median": 1e-3, "ef": 3}, 200_000, rng)
    assert clipped == 0
    assert np.median(values) == pytest.approx(1e-3, rel=0.02)
    assert np.percentile(values, 95) / np.median(values) == pytest.approx(3, rel=0.03)
    assert engine.distribution_mean({"dist": "lognormal", "median": 1e-3, "ef": 3}) == pytest.approx(values.mean(), rel=0.02)


def test_other_distributions_and_clipping():
    rng = np.random.default_rng(2)
    beta, _ = engine.sample_distribution({"dist": "beta", "a": 2, "b": 198}, 100_000, rng)
    assert beta.mean() == pytest.approx(0.01, rel=0.02)
    uni, _ = engine.sample_distribution({"dist": "uniform", "low": 0.1, "high": 0.2}, 10_000, rng)
    assert uni.min() >= 0.1 and uni.max() <= 0.2
    logu, _ = engine.sample_distribution({"dist": "loguniform", "low": 1e-4, "high": 1e-2}, 10_000, rng)
    assert logu.min() >= 1e-4 and logu.max() <= 1e-2
    assert np.median(logu) == pytest.approx(1e-3, rel=0.1)
    wide, clipped = engine.sample_distribution({"dist": "lognormal", "median": 0.5, "ef": 10}, 10_000, rng)
    assert clipped > 0 and wide.max() == 1.0


@pytest.mark.parametrize("spec", [
    {"dist": "lognormal", "median": 0, "ef": 3},
    {"dist": "lognormal", "median": 0.1, "ef": 0.5},
    {"dist": "beta", "a": 0, "b": 1},
    {"dist": "uniform", "low": 0.5, "high": 0.2},
    {"dist": "loguniform", "low": 0, "high": 0.2},
    {"dist": "normal", "mean": 0.1},
    {"dist": "beta", "a": True, "b": 1},
    {"dist": "beta", "a": float("nan"), "b": 1},
])
def test_invalid_distributions_are_rejected(spec):
    with pytest.raises(GuiError):
        engine.sample_distribution(spec, 10, np.random.default_rng(0))


def test_uncertainty_is_the_exact_analysis_of_the_joint_samples():
    tree = load("pressure_tank.json")
    dists = {"S": {"dist": "lognormal", "median": 0.01, "ef": 3},
             "K2": {"dist": "beta", "a": 1, "b": 3000}}
    out = strict_json(engine.analyze_uncertainty(tree, distributions=dists, n=500, seed=7))
    s = engine.sample_distribution(dists["S"], 500, np.random.default_rng([7, zlib.crc32(b"S")]))[0]
    k2 = engine.sample_distribution(dists["K2"], 500, np.random.default_rng([7, zlib.crc32(b"K2")]))[0]
    ref = analyze(tree, {"S": s, "K2": k2})
    np.testing.assert_allclose(out["Q"], ref["Q"], rtol=1e-12)
    stats = out["Q_stats"]
    assert stats["n"] == 500
    np.testing.assert_allclose([stats["p05"], stats["p50"], stats["p95"]], np.percentile(ref["Q"], [5, 50, 95]))
    assert stats["mean"] == pytest.approx(np.mean(ref["Q"]))
    crit = out["importance"]["S"]["criticality"]
    assert crit["p50"] == pytest.approx(np.percentile(ref["importance"]["S"]["criticality"], 50))
    assert set(out["inputs"]) == {"S", "K2"}
    assert out["probabilities"]["E3"]["mean"] == pytest.approx(np.mean(ref["probabilities"]["E3"]))


def test_uncertainty_is_reproducible_and_per_event_streams_are_independent():
    tree = load("pressure_tank.json")
    a = {"dist": "lognormal", "median": 0.01, "ef": 3}
    b = {"dist": "uniform", "low": 1e-4, "high": 1e-3}
    one = engine.analyze_uncertainty(tree, distributions={"S": a}, n=200, seed=3)
    two = engine.analyze_uncertainty(tree, distributions={"S": a, "K2": b}, n=200, seed=3)
    again = engine.analyze_uncertainty(tree, distributions={"S": a, "K2": b}, n=200, seed=3)
    other = engine.analyze_uncertainty(tree, distributions={"S": a, "K2": b}, n=200, seed=4)
    assert one["inputs"]["S"]["stats"] == two["inputs"]["S"]["stats"]
    assert two["Q"] == again["Q"]
    assert two["Q"] != other["Q"]


def test_given_sample_vectors_fix_the_sample_count():
    tree = load("fta4b.json")
    columns = engine.load_example("fta4b.json")["probs"]
    out = engine.analyze_uncertainty(tree, samples=columns, n=999)
    assert out["n"] == 15 and len(out["Q"]) == 15
    np.testing.assert_allclose(out["Q"], analyze(tree, {k: np.array(v) for k, v in columns.items()})["Q"])
    with pytest.raises(GuiError, match="same length"):
        engine.analyze_uncertainty(tree, samples={"1": [0.1, 0.2], "2": [0.1, 0.2, 0.3]})


def test_uncertainty_rejects_bad_requests():
    tree = load("pressure_tank.json")
    with pytest.raises(GuiError, match="no event"):
        engine.analyze_uncertainty(tree)
    with pytest.raises(GuiError):
        engine.analyze_uncertainty(tree, distributions={"X": {"dist": "beta", "a": 1, "b": 1}})
    with pytest.raises(GuiError):
        engine.analyze_uncertainty(tree, distributions={"S": {"dist": "beta", "a": 1, "b": 1}}, n=engine.MAX_SAMPLES + 1)
    with pytest.raises(GuiError):
        engine.analyze_uncertainty(tree, distributions={"S": {"dist": "beta", "a": 1, "b": 1}}, seed=-1)


def test_infinite_rrw_percentiles_are_null_not_nan():
    tree = {"id": "TOP", "gate": "AND", "children": [leaf("A", 0.1), leaf("B", 0.2)]}
    out = engine.analyze_uncertainty(tree, distributions={"A": {"dist": "uniform", "low": 0.05, "high": 0.2}}, n=50)
    rrw = strict_json(out)["importance"]["A"]["rrw"]
    assert rrw["p95"] is None and rrw["mean"] is None
    q = engine.quantiles([1.0, 2.0, math.inf, math.nan])
    assert q["n"] == 3 and q["p05"] == 1.0 and math.isinf(q["p95"])
    assert to_jsonable(q)["max"] is None


# ---------------------------------------------------------- probability file --
def b64(text):
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def test_uploaded_probability_files_use_the_engine_reader():
    out = engine.read_prob_upload("probs.csv", b64("A,B\n0.1,0.2\n0.3,0.4\n"))
    assert out == {"columns": {"A": [0.1, 0.3], "B": [0.2, 0.4]}, "rows": 2, "filename": "probs.csv"}
    ptbr = engine.read_prob_upload("PROBS.CSV", b64("A;B\n0,1;0,2\n"))
    assert ptbr["columns"] == {"A": [0.1], "B": [0.2]}
    with pytest.raises(GuiError, match="mine.csv:2"):
        engine.read_prob_upload("mine.csv", b64("A,B\n0.1,x\n"))
    with pytest.raises(GuiError, match="Unsupported"):
        engine.read_prob_upload("probs.txt", b64("A\n0.1\n"))
    with pytest.raises(GuiError, match="base64"):
        engine.read_prob_upload("probs.csv", "@@@")


# ------------------------------------------------------------------ server --
@pytest.fixture(scope="module")
def server():
    httpd = make_server("127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def call(url, body=None, headers=None, raw=None):
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.headers.get("Content-Type"), response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get("Content-Type"), error.read()


def test_server_serves_the_page_and_static_files(server):
    status, ctype, body = call(server + "/")
    assert status == 200 and ctype.startswith("text/html") and b'src="/static/app.js"' in body
    status, ctype, _ = call(server + "/static/app.js")
    assert status == 200 and ctype.startswith("text/javascript")
    for path in ("/static/../server.py", "/static/%2e%2e/engine.py", "/static/nope.js", "/nope"):
        assert call(server + path)[0] == 404


def test_server_json_endpoints(server):
    status, ctype, body = call(server + "/api/info")
    assert status == 200 and ctype.startswith("application/json")
    assert json.loads(body)["examples"] is True
    status, _, body = call(server + "/api/examples")
    assert "pressure_tank.json" in [e["file"] for e in json.loads(body)]
    status, _, body = call(server + "/api/examples/pressure_tank.json")
    tree = json.loads(body)["tree"]
    status, _, body = call(server + "/api/analyze", {"tree": tree})
    assert status == 200 and json.loads(body)["Q"] == pytest.approx(analyze(tree)["Q"])
    status, _, body = call(server + "/api/whatif", {"tree": tree, "forced": {"T": "failed"}})
    assert status == 200 and json.loads(body)["Q"] == pytest.approx(1.0)
    dists = {"S": {"dist": "lognormal", "median": 0.01, "ef": 3}}
    status, _, body = call(server + "/api/uncertainty", {"tree": tree, "distributions": dists, "n": 100, "seed": 1})
    assert status == 200 and len(json.loads(body)["Q"]) == 100
    status, _, body = call(server + "/api/probfile", {"filename": "p.csv", "content_b64": b64("S\n0.02\n")})
    assert status == 200 and json.loads(body)["columns"] == {"S": [0.02]}
    status, _, body = call(server + "/api/import", {"data": load("flat_tree.json")})
    assert status == 200 and json.loads(body)["format"] == "flat"


def test_server_reports_bad_input_as_400(server):
    for path, body in [("/api/analyze", {"tree": {"id": "T", "gate": "AND", "children": []}}),
                       ("/api/analyze", {}),
                       ("/api/whatif", {"tree": load("basic_tree.json"), "forced": {"X": "failed"}}),
                       ("/api/uncertainty", {"tree": load("basic_tree.json")})]:
        status, _, raw = call(server + path, body)
        assert status == 400, path
        assert json.loads(raw)["error"]
    status, _, raw = call(server + "/api/analyze", raw=b"{not json")
    assert status == 400 and "JSON" in json.loads(raw)["error"]
    status, _, raw = call(server + "/api/analyze", raw=b"[1]")
    assert status == 400
    assert call(server + "/api/nope", {})[0] == 404


def test_server_refuses_foreign_host_headers(server):
    status, _, raw = call(server + "/api/info", headers={"Host": "attacker.example:80"})
    assert status == 403
    assert call(server + "/api/info", headers={"Host": "localhost"})[0] == 200
    assert call(server + "/api/info", headers={"Host": "[::1]:8765"})[0] == 200


# --------------------------------------------------------------------- cli --
def test_cli_dispatches_gui_without_touching_other_flags(monkeypatch, capsys):
    from faultree import cli
    from faultree.gui import server as gui_server

    seen = {}
    monkeypatch.setattr(gui_server, "serve", lambda host, port, open_browser: seen.update(
        host=host, port=port, open_browser=open_browser) or 0)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["gui", "--port", "9123", "--no-browser"])
    assert exit_info.value.code == 0
    assert seen == {"host": "127.0.0.1", "port": 9123, "open_browser": False}
    cli.main([os.path.join(REPO, "examples", "basic_tree.json"), "--cut-sets", "2"])
    assert json.loads(capsys.readouterr().out)["complete"] is True


def test_python_m_faultree_gui_help():
    result = subprocess.run([sys.executable, "-m", "faultree", "gui", "--help"],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0
    assert "--no-browser" in result.stdout and "--port" in result.stdout


def test_serve_falls_back_to_a_free_port(monkeypatch):
    from faultree.gui import server as gui_server

    blocker = make_server("127.0.0.1", 0)
    monkeypatch.setattr(gui_server, "DEFAULT_PORT", blocker.server_address[1])
    started = {}

    def fake_forever(self):
        started["port"] = self.server_address[1]
        raise KeyboardInterrupt

    monkeypatch.setattr(gui_server.GuiServer, "serve_forever", fake_forever)
    try:
        assert gui_server.serve("127.0.0.1", None, open_browser=False) == 0
        assert started["port"] != blocker.server_address[1]
        # an explicitly requested port that is taken is an error, not a fallback
        assert gui_server.serve("127.0.0.1", blocker.server_address[1], open_browser=False) == 1
    finally:
        blocker.server_close()


# ------------------------------------------------------------ static files --
def test_every_interface_string_is_in_both_languages():
    dictionary = (STATIC_DIR / "i18n.js").read_text(encoding="utf-8")
    body = re.search(r"const D = (\{.*?\n\});", dictionary, re.S).group(1)
    entries = json.loads(re.sub(r",(\s*[}\]])", r"\1", body))
    for key, pair in entries.items():
        assert len(pair) == 2 and all(isinstance(x, str) and x for x in pair), key
    used = set()
    for path in STATIC_DIR.iterdir():
        text = path.read_text(encoding="utf-8")
        used |= set(re.findall(r'\bt\("([\w.]+)"', text))
        used |= set(re.findall(r'data-i18n(?:-title|-aria-label)?="([\w.]+)"', text))
    dynamic = {key for key in used if key.endswith(".")}
    missing = sorted(key for key in used - dynamic if key not in entries)
    assert not missing
    for prefix in dynamic:
        assert any(key.startswith(prefix) for key in entries), prefix


def test_page_loads_nothing_from_the_network():
    for path in STATIC_DIR.iterdir():
        text = path.read_text(encoding="utf-8")
        assert not re.search(r'(src|href)="https?://', text), path.name
        assert "fonts.googleapis" not in text and "cdn" not in text.lower(), path.name
    index = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    for ref in re.findall(r'(?:src|href)="/static/([^"]+)"', index):
        assert (STATIC_DIR / ref).is_file(), ref
