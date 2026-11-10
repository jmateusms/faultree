"""Bounded positive minimal cuts for monotone fault trees (not XOR implicants),
and their minimal path sets."""
from itertools import combinations
from typing import Any, Dict, FrozenSet, List, Set

from .builder import (_children_of, _gate_semantics, _is_leaf, _leaf_id, _walk, compile_tree,
                      normalize_tree, gather_nodes, collect_basic_events)
from .dynamic import is_pand


def minimal_cut_sets(tree: Dict[str, Any], *, max_order: int = 6,
                     max_candidates: int = 10_000, max_sets: int = 1_000,
                     max_basic_events: int = 32) -> Dict[str, Any]:
    """Enumerate increasing-size failure sets, with explicit completeness.

    Only AND, OR and at-least-k semantics are admitted. Limits bound the
    enumeration, not compilation time or BDD memory. A returned cut is always
    minimal, including when the result is truncated. No probabilities or rare-
    event approximation are used. Non-monotone implicants are a separate API.
    """
    for name, value, minimum in [("max_order", max_order, 0),
                                 ("max_candidates", max_candidates, 1),
                                 ("max_sets", max_sets, 1),
                                 ("max_basic_events", max_basic_events, 1)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    normalized = normalize_tree(tree)
    def combine(kind, k, parts):
        if kind not in {"AND", "OR", "AT_LEAST_K"}:
            raise ValueError("minimal cut sets require monotone AND/OR/K-of-N semantics; "
                             "XOR/signed implicants are not supported")
    registry = {}
    gather_nodes(normalized, registry)
    _walk(normalized, registry, False, lambda node: None, combine)
    ids = []
    collect_basic_events(normalized, ids)
    events = sorted(set(ids))
    if len(events) > max_basic_events:
        raise ValueError("number of basic events exceeds max_basic_events")
    compiled = compile_tree(normalized, include_expressions=False)
    cuts = []
    checked = 0
    reason = None
    covered = compiled.bdd.false
    for order in range(min(max_order, len(events)) + 1):
        for candidate_tuple in combinations(events, order):
            if checked >= max_candidates:
                reason = "max_candidates"
                break
            checked += 1
            candidate = frozenset(candidate_tuple)
            if any(cut.issubset(candidate) for cut in cuts):
                continue
            assignment = {event: event in candidate for event in events}
            if compiled.bdd.let(assignment, compiled.top) == compiled.bdd.true:
                cuts.append(candidate)
                function = compiled.bdd.true
                for event in candidate_tuple:
                    function &= compiled.bdd.var(event)
                covered |= function
                if covered == compiled.top:
                    break
                if len(cuts) >= max_sets:
                    reason = "max_sets"
                    break
        if reason or covered == compiled.top:
            break
    complete = covered == compiled.top
    if not complete and reason is None:
        reason = "max_order"
    return {"cut_sets": [sorted(cut) for cut in cuts], "complete": complete,
            "truncated": not complete, "reason": None if complete else reason,
            "candidates_checked": checked, "basic_events": events,
            "limits": {"max_order": max_order, "max_candidates": max_candidates,
                       "max_sets": max_sets, "max_basic_events": max_basic_events},
            "semantics": "positive minimal cuts of a monotone fault tree"}


#: Bounds of the path-set expansion: sets kept per family, and pairs of sets
#: joined per product, before the order limit is lowered to stay within them.
MAX_PAIRS = 200_000
#: Nodes the BDD may grow by while checking that a list is complete.
MAX_CHECK_NODES = 200_000


def minimal_path_sets(tree: Dict[str, Any], *, max_order: int = 10,
                      max_sets: int = 1_000, max_family: int = 2_000) -> Dict[str, Any]:
    """Minimal sets of events whose non-occurrence keeps the top event from
    occurring: in a reliability block diagram, the minimal sets of blocks
    whose working keeps the system working (its minimal paths).

    They are built top-down: an OR gate is kept from occurring by one set of
    each input (the product of their families), an AND or INHIBIT gate by one
    set of any input (the union), and an at-least-k gate by sets of n - k + 1
    of its n inputs; supersets are dropped at every step. A house event that
    has occurred cannot be prevented, and one that has not needs nothing. A
    conditioning event in a path set means that the condition does not hold;
    a PAND gate is one event. XOR is not monotone and is rejected.

    Sets longer than ``max_order`` are dropped. When a family would hold more
    than ``max_family`` sets, or a product more than ``MAX_PAIRS`` pairs, the
    order limit is lowered until it does not (``reason`` "max_family" or
    "max_pairs"), so the list stays complete up to the order in
    ``complete_to_order``. ``complete`` is true when the expansion left no set
    out; otherwise it is checked on the BDD (the listed paths cover every
    state in which the top event does not occur), within MAX_CHECK_NODES new
    nodes, past which the list counts as incomplete. Every listed set is
    minimal, whether or not the list is complete.
    """
    for name, value, minimum in [("max_order", max_order, 0), ("max_sets", max_sets, 1),
                                 ("max_family", max_family, 1)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    normalized = normalize_tree(tree)
    registry: Dict[str, Dict[str, Any]] = {}
    gather_nodes(normalized, registry)
    # limit: sets longer are left out; dropped: some set was left out (the
    # list may then be incomplete); cause: what last lowered the limit.
    state: Dict[str, Any] = {"limit": max_order, "dropped": False, "cause": None}
    memo: Dict[str, List[FrozenSet[str]]] = {}

    def lower(limit: int, cause: str) -> None:
        if limit < state["limit"]:
            state["limit"], state["dropped"], state["cause"] = limit, True, cause

    def minimal(family) -> List[FrozenSet[str]]:
        # Shortest first: a set is minimal unless a strictly shorter kept set
        # is inside it (two different sets of one length never are).
        out: List[FrozenSet[str]] = []
        length, shorter = -1, 0
        for s in sorted(set(family), key=len):
            if len(s) > state["limit"]:
                state["dropped"] = True
                break
            if len(s) != length:
                length, shorter = len(s), len(out)
            if not any(out[i] <= s for i in range(shorter)):
                out.append(s)
                if len(out) > max_family:
                    # Every shorter set is already in: keep those.
                    lower(len(s) - 1, "max_family")
                    return [t for t in out if len(t) <= state["limit"]]
        return out

    def product(a, b) -> List[FrozenSet[str]]:
        if len(a) * len(b) > MAX_PAIRS:
            limit = state["limit"]
            while limit > 0 and (sum(len(x) <= limit for x in a)
                                 * sum(len(y) <= limit for y in b)) > MAX_PAIRS:
                limit -= 1
            lower(limit, "max_pairs")
            a = [x for x in a if len(x) <= limit]
            b = [y for y in b if len(y) <= limit]
        return minimal(x | y for x in a for y in b)

    def at_least(m: int, families) -> List[FrozenSet[str]]:
        # chosen[j]: sets that prevent j of the inputs seen so far.
        chosen: List[List[FrozenSet[str]]] = [[frozenset()]] + [[] for _ in range(m)]
        for i, family in enumerate(families):
            for j in range(min(i + 1, m), 0, -1):
                chosen[j] = minimal(chosen[j] + product(chosen[j - 1], family))
        return chosen[m]

    active: Set[int] = set()

    def paths(node: Dict[str, Any]) -> List[FrozenSet[str]]:
        if "ref" in node:
            target = registry.get(str(node["ref"]))
            if target is None:
                raise ValueError(f"Unknown ref '{node['ref']}'")
            return paths(target)
        if _is_leaf(node) or is_pand(node):
            if node.get("event_type") == "house":
                return [] if node["state"] else [frozenset()]
            return [frozenset([_leaf_id(node)])]
        key = node.get("id")
        if key is not None and str(key) in memo:
            return memo[str(key)]
        if id(node) in active:
            raise ValueError(f"Cycle detected in tree at '{key}'")
        active.add(id(node))
        children = _children_of(node)
        if not children:
            raise ValueError(f"Gate '{node.get('gate')}' node '{key}' has no children")
        kind, k = _gate_semantics(node, len(children), False)
        families = [paths(child) for child in children]
        active.discard(id(node))
        if kind == "OR":
            out = [frozenset()]
            for family in families:
                out = product(out, family)
        elif kind == "AND":
            out = minimal(s for family in families for s in family)
        elif kind == "AT_LEAST_K":
            needed = len(children) - k + 1
            out = [] if needed > len(children) else at_least(needed, families)
        else:
            raise ValueError("minimal path sets require monotone AND/OR/K-of-N "
                             "semantics; XOR/signed implicants are not supported")
        if key is not None:
            memo[str(key)] = out
        return out

    family = paths(normalized)
    found = sorted((s for s in family if len(s) <= state["limit"]),
                   key=lambda x: (len(x), sorted(x)))
    if len(found) < len(family):
        state["dropped"] = True
    complete_to = state["limit"]
    trimmed = len(found) > max_sets
    if trimmed:
        complete_to = min(complete_to, len(found[max_sets]) - 1)
        found = [s for s in found if len(s) <= complete_to]
        state["dropped"], state["cause"] = True, state["cause"] or "max_sets"
    if not state["dropped"]:
        complete = True  # the expansion left nothing out
    elif trimmed:
        complete = False  # minimal path sets were cut from the list
    else:
        # Something was left out, perhaps only supersets: check on the BDD,
        # within a node budget (past it, the list is complete only to order).
        compiled = compile_tree(normalized, include_expressions=False)
        bdd, budget = compiled.bdd, len(compiled.bdd) + MAX_CHECK_NODES
        covered, complete = bdd.false, False
        for s in found:
            works = bdd.true
            for event in s:
                works &= ~bdd.var(event)
            covered |= works
            if len(bdd) > budget:
                break
        else:
            complete = covered == ~compiled.top
    reason = None if complete else state["cause"] or "max_order"
    return {"path_sets": [sorted(s) for s in found], "complete": complete,
            "truncated": not complete, "reason": reason,
            "complete_to_order": None if complete else complete_to,
            "limits": {"max_order": max_order, "max_sets": max_sets, "max_family": max_family},
            "semantics": "minimal path sets of a monotone fault tree"}
