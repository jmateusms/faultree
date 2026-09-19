"""Bounded positive minimal cuts for monotone fault trees (not XOR implicants)."""
from itertools import combinations
from typing import Any, Dict

from .builder import _walk, compile_tree, normalize_tree, gather_nodes, collect_basic_events


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
