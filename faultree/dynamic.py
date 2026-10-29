"""PRIORITY AND (PAND) gates, quantified from failure-time models.

A PAND gate occurs by the mission time t when all its inputs occur by t, in
order from left to right: T_1 < T_2 < ... < T_n <= t. A Boolean variable
records whether an event occurred, not when, so the static BDD cannot express
the gate. faultree quantifies each PAND as an independent module instead:

- every basic event under the gate has a time-to-failure model
  (``failure_model``) and the tree has a ``mission_time``;
- the inputs share no events with each other, and the gate's events appear
  nowhere else in the tree, so the inputs' failure times are independent of
  each other and of the rest of the tree.

Each input i has a failure-time CDF F_i(s) on a grid over [0, t]: from its own
model for a basic event, or for a static subtree from the BDD of that subtree
evaluated with every event's F(s) (the subtree is coherent, so it stays failed
once it fails and its CDF is its failure probability at s). The ordered CDF
follows by the recursion

    G_1 = F_1,    G_k(s) = integral over [0, s] of G_{k-1}(u) dF_k(u),

a Stieltjes integral computed with the trapezoidal rule, and P(PAND) = G_n(t).
The PAND then enters the main BDD as one variable with that probability, which
is exact because the module is independent of everything else.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
import math

import numpy as np

#: Grid intervals over [0, mission_time]; even, so the half grid is a subset.
DEFAULT_PAND_STEPS = 4096

#: Time-to-failure models a PAND input may use (the GUI's failure models
#: without ``binomial``, which counts demands, not time).
LIFETIME_PARAMS = {
    "exponential": ("lambda",),
    "weibull": ("beta", "eta"),
    "normal": ("mu", "sigma"),
    "lognormal": ("tmed", "s"),
    "gamma": ("alpha", "theta"),
    "uniform": ("a", "b"),
    "poisson": ("lambda", "k"),
}

#: Gates allowed inside a PAND module: coherent, so an input stays failed once
#: it fails and its failure time is well defined.
MODULE_GATES = ("AND", "OR", "K_OF_N", "PAND")


def is_pand(node: Dict[str, Any]) -> bool:
    return isinstance(node, dict) and node.get("gate") == "PAND"


# ------------------------------------------------------------ distributions --
def _lower_gamma_regularized(a: float, x: float) -> float:
    """P(a, x), the regularized lower incomplete gamma function (series for
    x < a + 1, Lentz continued fraction for the complement otherwise)."""
    if x <= 0.0:
        return 0.0
    if math.isinf(x):
        return 1.0
    log_front = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1.0:
        ap, term = a, 1.0 / a
        total = term
        for _ in range(10_000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return min(1.0, total * math.exp(log_front))
    tiny = 1e-300
    b = x + 1.0 - a
    c, d = 1.0 / tiny, 1.0 / b
    h = d
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = d if abs(d) > tiny else tiny
        c = b + an / c
        c = c if abs(c) > tiny else tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return max(0.0, 1.0 - math.exp(log_front) * h)


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    return np.array([0.5 * math.erfc(-v / math.sqrt(2.0)) for v in np.ravel(z)]).reshape(np.shape(z))


def lifetime_params(model: Any, event_id: str) -> Dict[str, float]:
    """Validate a time-to-failure model; return its numeric parameters."""
    if not isinstance(model, dict) or model.get("dist") not in LIFETIME_PARAMS:
        raise ValueError(
            f"Event '{event_id}' under a PAND gate needs a failure_model with a "
            "time-to-failure distribution (" + ", ".join(LIFETIME_PARAMS) + ")")
    dist = model["dist"]
    params: Dict[str, float] = {}
    for name in LIFETIME_PARAMS[dist]:
        value = model.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Event '{event_id}': {dist} failure model needs a finite '{name}'")
        params[name] = float(value)
    ok = {
        "exponential": lambda p: p["lambda"] >= 0,
        "weibull": lambda p: p["beta"] > 0 and p["eta"] > 0,
        "normal": lambda p: p["sigma"] > 0,
        "lognormal": lambda p: p["tmed"] > 0 and p["s"] > 0,
        "gamma": lambda p: p["alpha"] > 0 and p["theta"] > 0,
        "uniform": lambda p: 0 <= p["a"] < p["b"],
        "poisson": lambda p: p["lambda"] >= 0 and p["k"] >= 1 and p["k"] == int(p["k"]),
    }[dist]
    if not ok(params):
        raise ValueError(f"Event '{event_id}': invalid {dist} failure model parameters {params}")
    return params


def lifetime_cdf(model: Dict[str, Any], s: np.ndarray, event_id: str = "event") -> np.ndarray:
    """F(s) = P(T <= s) of a time-to-failure model, elementwise over times s >= 0.

    The formulas match the GUI's failure models, so F(mission_time) equals the
    ``prob`` the GUI saves. ``poisson`` is the time of the k-th failure of a
    Poisson process, P(N(s) >= k)."""
    p = lifetime_params(model, event_id)
    s = np.asarray(s, dtype=float)
    dist = model["dist"]
    if dist == "exponential":
        return -np.expm1(-p["lambda"] * s)
    if dist == "weibull":
        return -np.expm1(-np.power(s / p["eta"], p["beta"]))
    if dist == "normal":
        return _normal_cdf((s - p["mu"]) / p["sigma"])
    if dist == "lognormal":
        with np.errstate(divide="ignore"):
            z = np.where(s > 0, np.log(np.where(s > 0, s, 1.0) / p["tmed"]) / p["s"], -np.inf)
        return np.where(s > 0, _normal_cdf(np.where(np.isfinite(z), z, 0.0)), 0.0)
    if dist == "gamma":
        return np.array([_lower_gamma_regularized(p["alpha"], v / p["theta"]) for v in s])
    if dist == "uniform":
        return np.clip((s - p["a"]) / (p["b"] - p["a"]), 0.0, 1.0)
    # poisson: time of the k-th event
    return np.array([_lower_gamma_regularized(p["k"], p["lambda"] * v) for v in s])


def ordered_cdf(cdfs: List[np.ndarray]) -> np.ndarray:
    """G_n on the grid: P(T_1 < ... < T_n <= s) for independent T_i whose CDFs
    are given on one grid starting at s = 0.

    Trapezoidal Stieltjes sums: G_k[j] = G_k[j-1] + (G_{k-1}[j-1] + G_{k-1}[j])/2
    * (F_k[j] - F_k[j-1]). Probability mass at s <= 0 (a failed house event, a
    normal model with mass below zero) counts as failing at time 0; ties at
    time 0 are not ordered, so G_k[0] = 0 for k >= 2."""
    g = np.asarray(cdfs[0], dtype=float)
    for f in cdfs[1:]:
        f = np.asarray(f, dtype=float)
        increments = 0.5 * (g[1:] + g[:-1]) * np.diff(f)
        g = np.concatenate(([0.0], np.cumsum(increments)))
    return np.clip(g, 0.0, 1.0)


# ---------------------------------------------------------- module structure --
def _resolve(node: Dict[str, Any], registry: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if "ref" in node:
        return registry.get(str(node["ref"]))
    return node


def _children(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    children = node.get("children", [])
    return children if isinstance(children, list) else []


def _events_under(node: Dict[str, Any], registry: Dict[str, Dict[str, Any]],
                  stop: Optional[str] = None) -> Set[str]:
    """Basic-event ids reachable from ``node`` (refs followed, house events
    left out), not entering any occurrence of the gate with id ``stop``."""
    from .builder import _is_house, _is_leaf
    found: Set[str] = set()
    seen: Set[int] = set()
    stack = [node]
    while stack:
        current = _resolve(stack.pop(), registry)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if stop is not None and str(current.get("id")) == stop:
            continue
        if _is_leaf(current):
            if not _is_house(current):
                found.add(str(current.get("id")))
            continue
        stack.extend(_children(current))
    return found


def pand_gates(tree: Dict[str, Any], registry: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Distinct PAND gates of the tree, innermost first (post-order)."""
    order: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    def visit(node: Dict[str, Any], path: Set[str]) -> None:
        current = _resolve(node, registry)
        if current is None:
            return
        nid = str(current.get("id"))
        if nid in path:
            return  # the cycle is reported by the main traversal
        for child in _children(current):
            visit(child, path | {nid})
        if is_pand(current) and nid not in seen:
            seen.add(nid)
            order.append(current)

    visit(tree, set())
    return order


def module_members(tree: Dict[str, Any], registry: Dict[str, Dict[str, Any]]) -> Set[str]:
    """Ids of all nodes strictly inside some PAND gate (refs followed)."""
    inside: Set[str] = set()
    for gate in pand_gates(tree, registry):
        stack = list(_children(gate))
        while stack:
            current = _resolve(stack.pop(), registry)
            if current is None or "id" not in current:
                continue
            nid = str(current["id"])
            if nid in inside:
                continue
            inside.add(nid)
            stack.extend(_children(current))
    return inside


def check_pand(tree: Dict[str, Any], registry: Dict[str, Dict[str, Any]]) -> None:
    """Reject PAND gates that faultree cannot quantify exactly."""
    from .builder import _is_house, _is_leaf
    gates = pand_gates(tree, registry)
    if not gates:
        return
    mission = tree.get("mission_time")
    if (isinstance(mission, bool) or not isinstance(mission, (int, float))
            or not math.isfinite(mission) or mission <= 0):
        raise ValueError("A tree with a PAND gate needs a positive, finite "
                         "'mission_time' on its top event")
    for gate in gates:
        gid = str(gate.get("id"))
        inputs = _children(gate)
        if len(inputs) < 2:
            raise ValueError(f"PAND gate '{gid}' needs at least two inputs")
        # Everything inside: allowed gates, and events with a time model.
        stack = list(inputs)
        checked: Set[int] = set()
        while stack:
            current = _resolve(stack.pop(), registry)
            if current is None or id(current) in checked:
                continue
            checked.add(id(current))
            nid = current.get("id")
            if _is_leaf(current):
                if _is_house(current):
                    continue
                lifetime_params(current.get("failure_model"), str(nid))
                continue
            if current.get("gate") not in MODULE_GATES:
                raise ValueError(
                    f"Gate '{nid}' ({current.get('gate')}) cannot be inside PAND gate "
                    f"'{gid}': only {', '.join(MODULE_GATES)} keep a failure time defined")
            stack.extend(_children(current))
        # Independence: inputs disjoint, and no event shared with the rest.
        per_input = [_events_under(child, registry) for child in inputs]
        for i in range(len(per_input)):
            for j in range(i + 1, len(per_input)):
                shared = per_input[i] & per_input[j]
                if shared:
                    raise ValueError(
                        f"PAND gate '{gid}': inputs {i + 1} and {j + 1} share events "
                        f"({', '.join(sorted(shared))}); its inputs must be independent")
        inside = set().union(*per_input)
        outside = _events_under(tree, registry, stop=gid)
        shared = inside & outside
        if shared:
            raise ValueError(
                f"PAND gate '{gid}': events {', '.join(sorted(shared))} also appear "
                "outside the gate; a PAND must be an independent module")


# ------------------------------------------------------------- quantification --
@dataclass(frozen=True)
class PandResult:
    """Failure-space probabilities at the mission time.

    ``gates``: each PAND gate's probability (Richardson-extrapolated).
    ``inside``: every node strictly inside a PAND gate, from the same grid.
    ``details``: per gate, the value, an error estimate, the grid and time."""

    gates: Dict[str, float]
    inside: Dict[str, float]
    details: Dict[str, Dict[str, Any]]


def _module_pass(tree: Dict[str, Any], registry: Dict[str, Dict[str, Any]],
                 gates: List[Dict[str, Any]], grid: np.ndarray) -> Dict[str, np.ndarray]:
    """CDF on ``grid`` of every node inside or at a PAND gate."""
    from dd.autoref import BDD
    from .builder import _is_house, _is_leaf, gather_nodes, node_to_bdd, wmc
    inside = module_members(tree, registry)
    nodes: Dict[str, Dict[str, Any]] = {}
    gather_nodes(tree, nodes)
    events = sorted(nid for nid in inside if _is_leaf(nodes[nid]) and not _is_house(nodes[nid]))
    pand_ids = [str(g["id"]) for g in gates]
    bdd = BDD()
    bdd.declare(*events, *[g for g in pand_ids if g not in events])
    cdf: Dict[str, np.ndarray] = {e: lifetime_cdf(nodes[e]["failure_model"], grid, e) for e in events}
    for gate in gates:  # innermost first, so nested PAND CDFs exist when needed
        inputs = []
        for child in _children(gate):
            function = node_to_bdd(bdd, child, registry)
            inputs.append(np.broadcast_to(np.asarray(wmc(bdd, function, cdf), dtype=float), grid.shape))
        cdf[str(gate["id"])] = ordered_cdf(inputs)
    for nid in inside:
        if nid not in cdf:
            function = node_to_bdd(bdd, nodes[nid], registry)
            cdf[nid] = np.broadcast_to(np.asarray(wmc(bdd, function, cdf), dtype=float), grid.shape)
    return cdf


def quantify_pand(tree: Dict[str, Any], registry: Dict[str, Dict[str, Any]],
                  steps: int = DEFAULT_PAND_STEPS) -> PandResult:
    """Probabilities of every PAND gate and of every node inside one, at the
    tree's mission time.

    The grid has ``steps`` intervals; a second pass on half of them gives a
    Richardson-extrapolated value (the trapezoidal rule is second order) and
    ``abs_error_estimate`` = |P_steps - P_half| / 3, the estimated error of the
    full-grid value before extrapolation."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 4 or steps % 2:
        raise ValueError("pand_steps must be an even integer >= 4")
    gates = pand_gates(tree, registry)
    if not gates:
        return PandResult({}, {}, {})
    t = float(tree["mission_time"])
    fine = _module_pass(tree, registry, gates, np.linspace(0.0, t, steps + 1))
    coarse = _module_pass(tree, registry, gates, np.linspace(0.0, t, steps // 2 + 1))
    gate_ids = {str(g["id"]) for g in gates}
    values: Dict[str, float] = {}
    details: Dict[str, Dict[str, Any]] = {}
    for nid in fine:
        p_fine, p_coarse = float(fine[nid][-1]), float(coarse[nid][-1])
        values[nid] = min(1.0, max(0.0, (4.0 * p_fine - p_coarse) / 3.0))
        if nid in gate_ids:
            details[nid] = {"Q": values[nid], "abs_error_estimate": abs(p_fine - p_coarse) / 3.0,
                            "steps": steps, "mission_time": t}
    members = module_members(tree, registry)
    return PandResult(gates={g: values[g] for g in gate_ids},
                      inside={n: values[n] for n in members},
                      details=details)
