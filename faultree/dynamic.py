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
The grid is uniform where the events' CDFs change slowly and is halved
locally where one of them rises steeply (a failure time much shorter than
the mission time, or a narrow wear-out distribution).
The PAND then enters the main BDD as one variable with that probability, which
is exact because the module is independent of everything else.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import math

import numpy as np

#: Grid intervals over [0, mission_time]; even, so the half grid is a subset.
DEFAULT_PAND_STEPS = 4096
#: Local grid refinement (see _quadrature_grid): an interval of the coarse
#: grid is halved while an event's CDF rises by more than REFINE_RISE / steps
#: across it, at most MAX_REFINEMENTS times. A module may hold at most
#: MAX_GRID_VALUES event CDF values on its fine grid (events x points), or
#: twice what its uniform grid needs, whichever is larger. Intervals left
#: steep may add at most UNRESOLVED_TOLERANCE to the result's error.
REFINE_RISE = 16
MAX_REFINEMENTS = 64
MAX_GRID_VALUES = 30_000_000
UNRESOLVED_TOLERANCE = 1e-6

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


_ERFC = np.frompyfunc(math.erfc, 1, 1)


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * np.asarray(_ERFC(-np.asarray(z, dtype=float) / math.sqrt(2.0)), dtype=float)


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

    def visit(node: Dict[str, Any], path: List[str]) -> None:
        current = _resolve(node, registry)
        if current is None:
            return
        if "id" in current:
            nid = str(current["id"])
            # The main traversal treats a PAND as one event and does not look
            # inside it, so a cycle through a PAND is reported here.
            if nid in path:
                cycle = path[path.index(nid):] + [nid]
                raise ValueError(
                    "Cycle detected in tree (id revisited on the same path; a ref"
                    " cycle or a duplicate id): " + " -> ".join(cycle))
            path = path + [nid]
        for child in _children(current):
            visit(child, path)
        if is_pand(current) and str(current.get("id")) not in seen:
            seen.add(str(current.get("id")))
            order.append(current)

    visit(tree, [])
    return order


def _members_of(gate: Dict[str, Any], registry: Dict[str, Dict[str, Any]]) -> Set[str]:
    """Ids of the nodes strictly inside one PAND gate (refs followed)."""
    inside: Set[str] = set()
    anonymous: Set[int] = set()  # nodes without an id: no entry, but their inputs count
    stack = list(_children(gate))
    while stack:
        current = _resolve(stack.pop(), registry)
        if current is None:
            continue
        if "id" in current:
            nid = str(current["id"])
            if nid in inside:
                continue
            inside.add(nid)
        elif id(current) in anonymous:
            continue
        else:
            anonymous.add(id(current))
        stack.extend(_children(current))
    return inside


def module_members(tree: Dict[str, Any], registry: Dict[str, Dict[str, Any]]) -> Set[str]:
    """Ids of all nodes strictly inside some PAND gate (refs followed)."""
    inside: Set[str] = set()
    for gate in pand_gates(tree, registry):
        inside |= _members_of(gate, registry)
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


def _crossing_bound(rise: np.ndarray, groups: List[List[np.ndarray]]) -> float:
    """Bound on the trapezoid error of intervals the grid leaves unresolved.

    In one interval, the Stieltjes sum for G_k = int G_{k-1} dF_k is off by at
    most (rise of G_{k-1}) x (rise of F_k), and an input rises at most by the
    sum of its events' rises. ``rise`` is events x intervals; ``groups`` holds,
    for each PAND gate of the module, the event rows of each of its inputs."""
    total = 0.0
    for inputs in groups:
        sums = np.array([rise[rows].sum(axis=0) for rows in inputs])
        total += float(np.sum((sums.sum(axis=0) ** 2 - (sums ** 2).sum(axis=0)) / 2.0))
    return total


def _quadrature_grid(events: Dict[str, Dict[str, Any]], groups: List[List[List[str]]],
                     t: float, steps: int, gate_id: str
                     ) -> Tuple[np.ndarray, np.ndarray, float]:
    """The coarse grid of one PAND module on [0, t], its events' CDFs on it
    (events x points, rows in the order of ``events``), and a bound on the
    error of intervals left unresolved.

    The grid starts as steps/2 equal intervals. An interval is halved while
    some event's CDF rises by more than REFINE_RISE / steps across it: a
    uniform grid misses a CDF that rises within a few of its intervals, and
    the half grid misses it the same way, so the error estimate would not
    show it. An input that is a subtree rises at most as fast as the sum of
    its events' CDFs; what refining the events leaves shows in the estimate.

    An interval still steep after MAX_REFINEMENTS halvings, or at float
    resolution, stays: an integrable singularity at 0 (Weibull, gamma with a
    shape below 1) or a near-deterministic failure time costs little when
    the other inputs barely change there. Its possible error is bounded by
    :func:`_crossing_bound`; above UNRESOLVED_TOLERANCE it is an error."""
    ids = list(events)
    rows = {eid: k for k, eid in enumerate(ids)}
    index = [[np.array([rows[e] for e in inputs if e in rows], dtype=int) for inputs in gate]
             for gate in groups]

    def cdfs_at(s: np.ndarray) -> np.ndarray:
        out = np.empty((len(ids), s.size))
        for k, eid in enumerate(ids):
            out[k] = lifetime_cdf(events[eid]["failure_model"], s, eid)
        return out

    grid = np.linspace(0.0, t, steps // 2 + 1)
    tol = REFINE_RISE / steps
    budget = max(MAX_GRID_VALUES, 2 * len(ids) * (steps + 1))
    base = cdfs_at(grid)
    points, new_times, new_cdfs = grid.size, [], []
    # Intervals still to examine: their end times and the CDFs there.
    left_t, right_t, left_f, right_f = grid[:-1], grid[1:], base[:, :-1], base[:, 1:]
    unresolved = 0.0
    for level in range(MAX_REFINEMENTS + 1):
        rise = right_f - left_f
        steep = (rise > tol).any(axis=0)
        if not steep.any():
            break
        left_t, right_t = left_t[steep], right_t[steep]
        left_f, right_f, rise = left_f[:, steep], right_f[:, steep], rise[:, steep]
        mids = 0.5 * (left_t + right_t)
        split = (mids > left_t) & (mids < right_t)  # False at float resolution
        if level == MAX_REFINEMENTS:
            split[:] = False
        unresolved += _crossing_bound(rise[:, ~split], index)
        if not split.any():
            break
        left_t, right_t, mids = left_t[split], right_t[split], mids[split]
        left_f, right_f = left_f[:, split], right_f[:, split]
        mid_f = cdfs_at(mids)
        points += mids.size
        if len(ids) * (2 * points - 1) > budget:
            raise ValueError(
                f"PAND gate '{gate_id}': resolving the failure-time CDFs of its "
                f"{len(ids)} events needs more than {budget} values ({len(ids)} events x "
                f"{2 * points - 1} grid points, pand_steps={steps}); failure times are "
                "very short, or distributions very narrow, next to the mission time "
                "(a smaller pand_steps refines less)")
        new_times.append(mids)
        new_cdfs.append(mid_f)
        left_t, right_t = np.concatenate((left_t, mids)), np.concatenate((mids, right_t))
        left_f = np.concatenate((left_f, mid_f), axis=1)
        right_f = np.concatenate((mid_f, right_f), axis=1)
    if unresolved > UNRESOLVED_TOLERANCE:
        raise ValueError(
            f"PAND gate '{gate_id}': inputs fail within the same interval of time "
            f"too short for the grid (mission time {t:g}); the order of their failures "
            f"cannot be resolved (possible error up to {unresolved:.2g})")
    times = np.concatenate([grid] + new_times)
    order = np.argsort(times, kind="stable")
    return times[order], np.concatenate([base] + new_cdfs, axis=1)[:, order], unresolved


def _halved(values: np.ndarray, mids: np.ndarray) -> np.ndarray:
    """Interleave grid values with the values at the interval midpoints."""
    out = np.empty(2 * values.size - 1)
    out[0::2] = values
    out[1::2] = mids
    return out


def _module_pass(nodes: Dict[str, Dict[str, Any]], registry: Dict[str, Dict[str, Any]],
                 gates: List[Dict[str, Any]], members: Set[str], grid: np.ndarray,
                 cdf: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Values at the mission time (the grid's last point) of one module's PAND
    gates and inner nodes, from its events' CDFs on the grid."""
    from dd.autoref import BDD
    from .builder import node_to_bdd, wmc
    cdf = dict(cdf)
    gate_ids = [str(g["id"]) for g in gates]
    bdd = BDD()
    bdd.declare(*cdf, *[g for g in gate_ids if g not in cdf])
    for gate in gates:  # innermost first, so nested PAND CDFs exist when needed
        inputs = []
        for child in _children(gate):
            function = node_to_bdd(bdd, child, registry)
            inputs.append(np.broadcast_to(np.asarray(wmc(bdd, function, cdf), dtype=float), grid.shape))
        cdf[str(gate["id"])] = ordered_cdf(inputs)
    at_t = {nid: float(values[-1]) for nid, values in cdf.items()}
    out = {nid: at_t[nid] for nid in gate_ids}
    for nid in members:
        if nid not in out:
            out[nid] = at_t[nid] if nid in at_t else float(
                wmc(bdd, node_to_bdd(bdd, nodes[nid], registry), at_t))
    return out


def quantify_pand(tree: Dict[str, Any], registry: Dict[str, Dict[str, Any]],
                  steps: int = DEFAULT_PAND_STEPS) -> PandResult:
    """Probabilities of every PAND gate and of every node inside one, at the
    tree's mission time.

    Each outermost PAND gate and everything inside it (a module) gets its own
    grid: ``steps`` intervals, more where one of its events' CDFs rises
    steeply (see :func:`_quadrature_grid`). A second pass on every other point
    gives a Richardson-extrapolated value (the trapezoidal rule is second
    order) and ``abs_error_estimate`` = |P_fine - P_coarse| / 3, the estimated
    error of the fine-grid value before extrapolation."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 4 or steps % 2:
        raise ValueError("pand_steps must be an even integer >= 4")
    gates = pand_gates(tree, registry)
    if not gates:
        return PandResult({}, {}, {})
    from .builder import _is_house, _is_leaf, gather_nodes
    nodes: Dict[str, Dict[str, Any]] = {}
    gather_nodes(tree, nodes)
    t = float(tree["mission_time"])
    inside = module_members(tree, registry)
    values: Dict[str, float] = {}
    details: Dict[str, Dict[str, Any]] = {}
    for outer in (g for g in gates if str(g["id"]) not in inside):
        oid = str(outer["id"])
        members = _members_of(outer, registry)
        module_gates = [g for g in gates if str(g["id"]) in members or str(g["id"]) == oid]
        events = {nid: nodes[nid] for nid in sorted(members)
                  if _is_leaf(nodes[nid]) and not _is_house(nodes[nid])}
        groups = [[sorted(_events_under(child, registry)) for child in _children(g)]
                  for g in module_gates]
        coarse_grid, coarse_cdf, unresolved = _quadrature_grid(events, groups, t, steps, oid)
        mids = 0.5 * (coarse_grid[:-1] + coarse_grid[1:])
        fine_grid = _halved(coarse_grid, mids)
        fine_cdf = {eid: _halved(coarse_cdf[k], lifetime_cdf(events[eid]["failure_model"], mids, eid))
                    for k, eid in enumerate(events)}
        del coarse_cdf  # the coarse pass reads every other point of the fine arrays
        fine = _module_pass(nodes, registry, module_gates, members, fine_grid, fine_cdf)
        coarse = _module_pass(nodes, registry, module_gates, members, fine_grid[::2],
                              {eid: f[::2] for eid, f in fine_cdf.items()})
        del fine_cdf
        for nid, p_fine in fine.items():
            values[nid] = min(1.0, max(0.0, (4.0 * p_fine - coarse[nid]) / 3.0))
        for gate in module_gates:
            gid = str(gate["id"])
            details[gid] = {"Q": values[gid],
                            "abs_error_estimate": abs(fine[gid] - coarse[gid]) / 3.0 + unresolved,
                            "steps": steps, "grid_points": int(fine_grid.size),
                            "mission_time": t}
    return PandResult(gates={str(g["id"]): values[str(g["id"])] for g in gates},
                      inside={n: values[n] for n in inside},
                      details=details)
