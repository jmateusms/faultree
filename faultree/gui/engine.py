"""Analysis helpers behind the local GUI: plain dicts in, plain dicts out.

Nothing here speaks HTTP, so every endpoint of :mod:`faultree.gui.server` can
be tested by calling these functions. They only prepare inputs for, and
summarize the output of, :func:`faultree.analyze` and
:func:`faultree.minimal_cut_sets`; the analysis semantics are the library's.
"""
from __future__ import annotations

import base64
import binascii
import json
import math
import os
import tempfile
import zlib
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ..builder import (
    _is_leaf,
    analyze,
    collect_basic_events,
    compile_tree,
    gather_nodes,
    load_probs_from_file,
    normalize_tree,
    quantify_compiled,
    to_jsonable,
    var_probs_from_tree,
)
from ..cutsets import minimal_cut_sets

#: Upper bound on the number of joint samples one uncertainty run may draw.
MAX_SAMPLES = 50_000
#: Default number of joint samples.
DEFAULT_SAMPLES = 2_000
#: The BDD is drawn only up to this many decision nodes.
BDD_DRAW_LIMIT = 60
#: Percentiles reported for every sampled quantity.
PERCENTILES = (5, 50, 95)
#: Bins of the input-distribution previews.
PREVIEW_BINS = 30
#: Example trees whose probabilities live in a separate file (and the file
#: is not named by a top-level ``prob_file``).
EXAMPLE_PROBS = {"large_tree.json": "large_example_probs.csv"}
#: Two-sided 90 % standard normal quantile: an error factor is p95 / p50.
Z95 = 1.6448536269514722


class GuiError(ValueError):
    """Bad input to a GUI endpoint (reported to the browser as HTTP 400)."""


# ----------------------------------------------------------------- examples --
def examples_dir() -> Optional[Path]:
    """The examples directory: the checkout's ``examples/`` (source and
    editable installs) or the copy packaged as ``faultree/examples``."""
    package = Path(__file__).resolve().parent.parent
    for candidate in (package.parent / "examples", package / "examples"):
        if candidate.is_dir() and any(candidate.glob("*.json")):
            return candidate
    return None


def _example_path(name: str) -> Path:
    folder = examples_dir()
    if folder is None:
        raise GuiError("no examples directory found")
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(".json"):
        raise GuiError(f"unknown example: {name!r}")
    path = folder / name
    if not path.is_file():
        raise GuiError(f"unknown example: {name!r}")
    return path


def _count_nodes(tree: Dict[str, Any]) -> Tuple[int, int]:
    nodes: Dict[str, Dict[str, Any]] = {}
    gather_nodes(tree, nodes)
    basic = sum(1 for node in nodes.values() if _is_leaf(node))
    return basic, len(nodes) - basic


def list_examples() -> List[Dict[str, Any]]:
    """One summary per example tree; files the engine rejects are skipped."""
    folder = examples_dir()
    if folder is None:
        return []
    out = []
    for path in sorted(folder.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            tree = normalize_tree(raw)
            basic, gates = _count_nodes(tree)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        out.append({
            "file": path.name,
            "title": str(tree.get("name") or tree.get("id")),
            "title_pt": tree.get("name_pt"),
            "format": "flat" if "ft_nodes" in raw else "recursive",
            "basic_events": basic,
            "gates": gates,
            "success_mode": path.stem.endswith("_success"),
            "prob_file": raw.get("prob_file") or EXAMPLE_PROBS.get(path.name),
        })
    return out


def load_example(name: str) -> Dict[str, Any]:
    """An example ready for the editor, with its probability file applied."""
    path = _example_path(name)
    raw = json.loads(path.read_text(encoding="utf-8"))
    result = import_tree(raw)
    prob_file = raw.get("prob_file") or EXAMPLE_PROBS.get(name)
    if prob_file:
        columns = load_probs_from_file(str(path.parent / str(prob_file)))
        result["probs"] = {key: value.tolist() for key, value in columns.items()}
        result["prob_file"] = str(prob_file)
    result["file"] = name
    result["success_mode"] = path.stem.endswith("_success")
    return result


# -------------------------------------------------------------------- input --
def import_tree(data: Any) -> Dict[str, Any]:
    """Normalize a recursive or flat tree (or an API request body holding a
    ``tree`` and optional ``probs``) into the recursive form the editor uses."""
    probs = None
    if (isinstance(data, dict) and isinstance(data.get("tree"), dict)
            and not {"id", "gate", "ft_nodes"} & set(data)):
        probs = data.get("probs")
        data = data["tree"]
    if not isinstance(data, dict):
        raise GuiError("the tree must be a JSON object")
    fmt = "flat" if "ft_nodes" in data and "be_nodes" in data else "recursive"
    tree = normalize_tree(data)
    if "id" not in tree:
        raise GuiError("the top event has no 'id'")
    result: Dict[str, Any] = {"tree": tree, "format": fmt}
    if isinstance(probs, dict):
        result["probs"] = probs
    if fmt == "recursive" and data.get("prob_file"):
        result["prob_file_missing"] = str(data["prob_file"])
    return result


def read_prob_upload(filename: str, content_b64: str) -> Dict[str, Any]:
    """Parse an uploaded CSV/Excel probability file with the engine's reader."""
    if not isinstance(filename, str) or not filename:
        raise GuiError("missing file name")
    suffix = Path(filename).suffix.lower()
    if suffix not in (".csv", ".xlsx", ".xls"):
        raise GuiError("Unsupported file format. Use CSV or Excel.")
    try:
        content = base64.b64decode(content_b64 or "", validate=True)
    except (binascii.Error, ValueError, TypeError):
        raise GuiError("the file content is not valid base64") from None
    with tempfile.TemporaryDirectory(prefix="faultree-gui-") as folder:
        path = os.path.join(folder, "upload" + suffix)
        with open(path, "wb") as handle:
            handle.write(content)
        try:
            columns = load_probs_from_file(path)
        except ValueError as error:
            raise GuiError(str(error).replace(path, filename)) from None
    values = {str(key): np.asarray(value, dtype=float) for key, value in columns.items()}
    rows = max((value.size for value in values.values()), default=0)
    return {"columns": {key: value.tolist() for key, value in values.items()},
            "rows": int(rows), "filename": filename}


def _basic_ids(tree: Dict[str, Any]) -> List[str]:
    ids: List[str] = []
    collect_basic_events(tree, ids)
    return list(dict.fromkeys(ids))


def _flag(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise GuiError(f"'{name}' must be true or false")
    return value


def _int(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise GuiError(f"'{name}' must be an integer from {low} to {high}")
    return value


def _require_scalars(tree: Dict[str, Any]) -> None:
    vectors = [eid for eid, value in var_probs_from_tree(tree).items()
               if np.ndim(value) > 0]
    if vectors:
        raise GuiError("the point analysis needs one probability per event; "
                       "sample vectors go to the uncertainty analysis ("
                       + ", ".join(vectors) + ")")


# ------------------------------------------------------------------- output --
def bdd_graph(compiled: Any, probs: Mapping[str, Any], limit: int = BDD_DRAW_LIMIT
              ) -> Optional[Dict[str, Any]]:
    """The top-event BDD without complemented edges, for drawing.

    dd stores complemented edges; a reader expects a plain reduced BDD whose
    edges point to the 0 and 1 terminals, so complements are pushed down
    (one drawn node per distinct sub-function). Each node carries the
    probability of its sub-function, i.e. the Shannon recursion
    ``p(f) = q * p(f_high) + (1 - q) * p(f_low)`` that ``wmc`` evaluates.
    Returns ``None`` above ``limit`` decision nodes.
    """
    bdd = compiled.bdd
    nodes: Dict[int, Dict[str, Any]] = {}

    class TooBig(Exception):
        pass

    def rec(u: Any) -> Tuple[str, float]:
        if u == bdd.true:
            return "1", 1.0
        if u == bdd.false:
            return "0", 0.0
        key = int(u)
        entry = nodes.get(key)
        if entry is not None:
            return entry["id"], entry["p"]
        if len(nodes) >= limit:
            raise TooBig
        hi, lo = u.high, u.low
        if u.negated:
            hi, lo = ~hi, ~lo
        entry = {"id": f"n{len(nodes)}", "var": u.var, "level": int(u.level)}
        nodes[key] = entry
        entry["high"], p_high = rec(hi)
        entry["low"], p_low = rec(lo)
        q = float(probs.get(u.var, 0.0))
        entry["q"] = q
        entry["p"] = q * p_high + (1.0 - q) * p_low
        return entry["id"], entry["p"]

    try:
        root, p_root = rec(compiled.top)
    except TooBig:
        return None
    return {"root": root, "p": p_root, "nodes": list(nodes.values()),
            "levels": list(compiled.ordering)}


def _cut_set_summary(tree: Dict[str, Any], probs: Mapping[str, float], q: float,
                     max_order: int) -> Dict[str, Any]:
    try:
        found = minimal_cut_sets(tree, max_order=max_order)
    except ValueError as error:
        return {"available": False, "error": str(error)}
    rows = []
    for events in found["cut_sets"]:
        p = float(np.prod([float(probs[e]) for e in events])) if events else 1.0
        rows.append({"events": events, "order": len(events), "p": p,
                     "share": p / q if q > 0 else None})
    rows.sort(key=lambda row: (-row["p"], row["order"], row["events"]))
    ps = [row["p"] for row in rows]
    by_order: Dict[int, int] = {}
    for row in rows:
        by_order[row["order"]] = by_order.get(row["order"], 0) + 1
    return {
        "available": True,
        "rows": rows,
        "complete": found["complete"],
        "truncated": found["truncated"],
        "reason": found["reason"],
        "candidates_checked": found["candidates_checked"],
        "limits": found["limits"],
        "by_order": {str(k): v for k, v in sorted(by_order.items())},
        # Classic approximations, for comparison with the exact Q.
        "rare_event": float(sum(ps)),
        "mcub": float(1.0 - np.prod([1.0 - p for p in ps])) if ps else 0.0,
    }


def analyze_point(tree: Any, *, success_mode: bool = False, max_order: int = 6,
                  bdd_limit: int = BDD_DRAW_LIMIT) -> Dict[str, Any]:
    """Exact point analysis: Q (or R), every event, importance, cut sets, BDD."""
    started = perf_counter()
    success_mode = _flag(success_mode, "success_mode")
    max_order = _int(max_order, "max_order", 1, 12)
    tree = normalize_tree(tree)
    _require_scalars(tree)
    result = analyze(tree, None, success_mode=success_mode)
    top_id = str(tree["id"])
    inputs = {eid: float(p) for eid, p in var_probs_from_tree(tree).items()}
    q = float(result["Q"])
    compiled = compile_tree(tree, success_mode=success_mode, include_expressions=False)
    out = {
        "mode": "success" if success_mode else "failure",
        "top_id": top_id,
        "Q": q,
        "inputs": inputs,
        "basic_events": _basic_ids(tree),
        "probabilities": result["probabilities"],
        "conditional_Q": result["conditional_Q"],
        "birnbaum": result["birnbaum"],
        "importance": result["importance"],
        "expression": result["expression"],
        "symbolic": result["symbolic"],
        "assumptions": result["assumptions"],
        "bdd": {"nodes": len(compiled.bdd), "top_nodes": int(compiled.top.dag_size),
                "ordering": list(compiled.ordering),
                "graph": bdd_graph(compiled, inputs, bdd_limit)},
    }
    if success_mode:
        out["cut_sets"] = {"available": False, "error": "success_mode"}
    else:
        out["cut_sets"] = _cut_set_summary(tree, inputs, q, max_order)
    out["elapsed_ms"] = (perf_counter() - started) * 1000.0
    return to_jsonable(out)


def what_if(tree: Any, forced: Any, *, success_mode: bool = False) -> Dict[str, Any]:
    """Top-event and intermediate probabilities with some basic events set
    failed or working (``forced``: event id -> "failed" | "working").

    With independent basic events, conditioning on an event's state is the
    same as setting its probability to 1 or 0, which is how ``analyze``
    obtains ``conditional_Q``. In success mode the inputs are reliabilities,
    so a failed event has reliability 0.
    """
    success_mode = _flag(success_mode, "success_mode")
    tree = normalize_tree(tree)
    _require_scalars(tree)
    if not isinstance(forced, dict):
        raise GuiError("'forced' must map event ids to 'failed' or 'working'")
    basic = set(_basic_ids(tree))
    compiled = compile_tree(tree, success_mode=success_mode, include_expressions=False)
    base = var_probs_from_tree(tree)
    probs = dict(base)
    for eid, state in forced.items():
        if eid not in basic:
            raise GuiError(f"'{eid}' is not a basic event of this tree")
        if state not in ("failed", "working"):
            raise GuiError(f"state of '{eid}' must be 'failed' or 'working'")
        failed = state == "failed"
        probs[eid] = (0.0 if failed else 1.0) if success_mode else (1.0 if failed else 0.0)
    top_id = str(compiled.tree["id"])
    baseline = quantify_compiled(compiled, base)
    scenario = quantify_compiled(compiled, probs)
    return to_jsonable({
        "mode": "success" if success_mode else "failure",
        "top_id": top_id,
        "forced": forced,
        "Q": scenario[top_id],
        "base_Q": baseline[top_id],
        "probabilities": scenario,
        "base_probabilities": baseline,
    })


# -------------------------------------------------------------- uncertainty --
def _check_params(eid: str, spec: Mapping[str, Any]) -> Tuple[str, Dict[str, float]]:
    if not isinstance(spec, Mapping):
        raise GuiError(f"distribution of '{eid}' must be an object")
    kind = spec.get("dist")
    names = {"lognormal": ("median", "ef"), "beta": ("a", "b"),
             "uniform": ("low", "high"), "loguniform": ("low", "high")}
    if kind not in names:
        raise GuiError(f"'{eid}': unknown distribution {kind!r} "
                       "(use lognormal, beta, uniform or loguniform)")
    params: Dict[str, float] = {}
    for name in names[kind]:
        value = spec.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise GuiError(f"'{eid}': {kind} needs a finite number for '{name}'")
        params[name] = float(value)
    bad = None
    if kind == "lognormal":
        if not 0 < params["median"] <= 1:
            bad = "the median must be in (0, 1]"
        elif params["ef"] < 1:
            bad = "the error factor must be >= 1"
    elif kind == "beta":
        if params["a"] <= 0 or params["b"] <= 0:
            bad = "a and b must be > 0"
    elif kind == "uniform":
        if not 0 <= params["low"] <= params["high"] <= 1:
            bad = "0 <= low <= high <= 1 is required"
    elif not 0 < params["low"] <= params["high"] <= 1:
        bad = "0 < low <= high <= 1 is required"
    if bad:
        raise GuiError(f"'{eid}' ({kind}): {bad}")
    return kind, params


def event_rng(seed: int, event_id: str) -> np.random.Generator:
    """The random stream of one event: ``default_rng([seed, crc32(id)])``.

    Each event has its own stream, so adding, removing or reordering events
    leaves the samples of the others unchanged."""
    return np.random.default_rng([int(seed), zlib.crc32(event_id.encode("utf-8"))])


def sample_distribution(spec: Mapping[str, Any], n: int, rng: np.random.Generator,
                        event_id: str = "event") -> Tuple[np.ndarray, int]:
    """Draw ``n`` probabilities from one distribution; return them and how
    many lognormal draws above 1 were clipped to 1."""
    kind, p = _check_params(event_id, spec)
    clipped = 0
    if kind == "lognormal":
        sigma = math.log(p["ef"]) / Z95
        values = p["median"] * np.exp(sigma * rng.standard_normal(n))
        clipped = int(np.count_nonzero(values > 1.0))
        values = np.minimum(values, 1.0)
    elif kind == "beta":
        values = rng.beta(p["a"], p["b"], n)
    elif kind == "uniform":
        values = rng.uniform(p["low"], p["high"], n)
    else:
        values = np.exp(rng.uniform(math.log(p["low"]), math.log(p["high"]), n))
    return values, clipped


def distribution_mean(spec: Mapping[str, Any]) -> float:
    """Mean of a distribution as given (before clipping)."""
    kind, p = _check_params("event", spec)
    if kind == "lognormal":
        sigma = math.log(p["ef"]) / Z95
        return p["median"] * math.exp(sigma * sigma / 2)
    if kind == "beta":
        return p["a"] / (p["a"] + p["b"])
    if kind == "uniform":
        return (p["low"] + p["high"]) / 2
    if p["low"] == p["high"]:
        return p["low"]
    return (p["high"] - p["low"]) / math.log(p["high"] / p["low"])


def quantiles(values: Any) -> Dict[str, Any]:
    """Mean, standard deviation and 5/50/95 % percentiles of a sample.

    NaN entries (an undefined 0/0 ratio) are left out; infinite ones (RRW
    when one event guards the top event alone) are kept and make the upper
    percentiles infinite (reported as null), without interpolating across
    them."""
    x = np.asarray(values, dtype=float).ravel()
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {"n": 0}
    finite = bool(np.all(np.isfinite(x)))
    if finite:
        pct = np.percentile(x, PERCENTILES)
    else:  # inverted CDF: an actual sample, no interpolation towards inf
        ordered = np.sort(x)
        pct = [ordered[max(0, math.ceil(level / 100 * ordered.size) - 1)] for level in PERCENTILES]
    out = {"n": int(x.size), "min": float(x.min()), "max": float(x.max()),
           "mean": float(x.mean()) if finite else float("inf"),
           "std": float(x.std(ddof=1)) if finite and x.size > 1 else None}
    for level, value in zip(PERCENTILES, pct):
        out[f"p{level:02d}"] = float(value)
    return out


def _preview(values: np.ndarray, kind: Optional[str]) -> Dict[str, Any]:
    lo, hi = float(values.min()), float(values.max())
    log = kind in ("lognormal", "loguniform") and lo > 0 and hi / lo > 20
    if hi <= lo:
        return {"log": False, "edges": [lo, hi], "counts": [int(values.size)]}
    if log:
        edges = np.geomspace(lo, hi, PREVIEW_BINS + 1)
    else:
        edges = np.linspace(lo, hi, PREVIEW_BINS + 1)
    counts, _ = np.histogram(values, edges)
    return {"log": log, "edges": edges.tolist(), "counts": counts.tolist()}


def analyze_uncertainty(tree: Any, *, distributions: Any = None, samples: Any = None,
                        n: int = DEFAULT_SAMPLES, seed: int = 0,
                        success_mode: bool = False) -> Dict[str, Any]:
    """Propagate input uncertainty through the exact analysis.

    Events with a distribution are sampled (``n`` draws each, see
    :func:`event_rng`); events with given sample vectors (``samples``, e.g.
    from a probability file) use them, and all vectors must have the same
    length; the remaining events keep their point probability. Equal-length
    vectors are joint samples: draw ``i`` of every event forms scenario
    ``i``, which :func:`faultree.analyze` evaluates elementwise.
    """
    started = perf_counter()
    success_mode = _flag(success_mode, "success_mode")
    seed = _int(seed, "seed", 0, 2**32 - 1)
    tree = normalize_tree(tree)
    basic = _basic_ids(tree)
    distributions = distributions or {}
    samples = samples or {}
    if not isinstance(distributions, dict) or not isinstance(samples, dict):
        raise GuiError("'distributions' and 'samples' must be objects keyed by event id")
    for eid in list(distributions) + list(samples):
        if eid not in basic:
            raise GuiError(f"'{eid}' is not a basic event of this tree")
    given: Dict[str, np.ndarray] = {}
    tree_vectors = {eid: value for eid, value in var_probs_from_tree(tree).items()
                    if np.ndim(value) > 0}
    for eid, vector in {**tree_vectors, **samples}.items():
        if eid in distributions:
            continue  # an explicit distribution replaces the vector
        arr = np.asarray(vector, dtype=float)
        if arr.ndim != 1 or arr.size == 0:
            raise GuiError(f"samples of '{eid}' must be a non-empty list of numbers")
        given[eid] = arr
    lengths = {arr.size for arr in given.values()}
    if len(lengths) > 1:
        detail = ", ".join(f"{eid}={arr.size}" for eid, arr in given.items())
        raise GuiError("sample vectors must have the same length to be joint samples; got "
                       + detail)
    if lengths:
        n = lengths.pop()
    n = _int(n, "n", 2, MAX_SAMPLES)
    if not distributions and not given:
        raise GuiError("no event has a distribution or sample vector")

    probs: Dict[str, Any] = {eid: p for eid, p in var_probs_from_tree(tree).items()
                             if np.ndim(p) == 0}
    probs.update(given)
    clipped: Dict[str, int] = {}
    inputs: Dict[str, Any] = {}
    for eid in basic:
        kind = None
        if eid in distributions:
            spec = distributions[eid]
            values, clip = sample_distribution(spec, n, event_rng(seed, eid), eid)
            probs[eid] = values
            kind = spec.get("dist")
            if clip:
                clipped[eid] = clip
        elif eid in given:
            values = given[eid]
            kind = "samples"
        else:
            continue
        inputs[eid] = {"kind": kind, "stats": quantiles(values),
                       "preview": _preview(np.asarray(values, dtype=float), kind)}

    result = analyze(tree, probs, success_mode=success_mode)
    top_id = str(tree["id"])
    q = np.broadcast_to(np.asarray(result["Q"], dtype=float), (n,))
    nodes = {nid: quantiles(np.broadcast_to(np.asarray(v, dtype=float), (n,)))
             for nid, v in result["probabilities"].items()}
    importance: Dict[str, Any] = {}
    for eid in basic:
        measures = {"birnbaum": result["birnbaum"][eid]}
        if result["importance"] is not None:
            measures.update({key: value for key, value in result["importance"][eid].items()
                             if key != "birnbaum"})
        importance[eid] = {key: quantiles(np.broadcast_to(np.asarray(value, dtype=float), (n,)))
                           for key, value in measures.items()}
    return to_jsonable({
        "mode": "success" if success_mode else "failure",
        "top_id": top_id,
        "n": n,
        "seed": seed,
        "Q": q.tolist(),
        "Q_stats": quantiles(q),
        "probabilities": nodes,
        "importance": importance,
        "inputs": inputs,
        "clipped": clipped,
        "rng": "numpy.random.default_rng([seed, zlib.crc32(event_id)]) per event",
        "assumptions": result["assumptions"],
        "elapsed_ms": (perf_counter() - started) * 1000.0,
    })
