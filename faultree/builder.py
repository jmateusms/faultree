from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import os

try:
    from dd.autoref import BDD, Function
except Exception as e:
    raise RuntimeError("dd package is required. Install with `pip install dd`.") from e


def _is_leaf(node: Dict[str, Any]) -> bool:
    """A node is a basic/undeveloped (leaf) event iff it has no children and is
    marked basic either by event_type or by gate."""
    if node.get("children", []):
        return False
    et = node.get("event_type")
    gate = node.get("gate")
    return et in ("basic", "undeveloped") or gate in (None, "BASIC")


def _children_of(node: Any) -> List[Dict[str, Any]]:
    """Validated access to a node's children (input errors must surface as
    ValueError so the CLI/server can report them as bad input)."""
    if not isinstance(node, dict):
        raise ValueError(f"Tree node must be a JSON object, got: {node!r}")
    children = node.get("children", [])
    if not isinstance(children, list):
        raise ValueError(
            f"'children' of node '{node.get('id')}' must be a list, got: {children!r}")
    for ch in children:
        if not isinstance(ch, dict):
            raise ValueError(
                f"Child of node '{node.get('id')}' must be a JSON object, got: {ch!r}")
    return children


def _leaf_id(node: Dict[str, Any]) -> str:
    if "id" not in node:
        raise ValueError(f"Basic event node has no 'id' field: {node!r}")
    return str(node["id"])


def collect_basic_events(node: Dict[str, Any], vars_out: List[str]) -> None:
    # ref nodes are skipped: every ref target must also appear literally
    # somewhere in the tree, so its id is collected on that occurrence.
    if not isinstance(node, dict):
        raise ValueError(f"Tree node must be a JSON object, got: {node!r}")
    if "ref" in node:
        return
    if _is_leaf(node):
        vars_out.append(_leaf_id(node))
        return
    for ch in _children_of(node):
        collect_basic_events(ch, vars_out)


def build_manager(tree: Dict[str, Any], ordering: Optional[List[str]] = None) -> BDD:
    vars_out: List[str] = []
    collect_basic_events(tree, vars_out)
    needed = list(dict.fromkeys(vars_out))
    if ordering is None:
        ordering = needed
    else:
        if len(set(ordering)) != len(ordering):
            dups = sorted({v for v in ordering if ordering.count(v) > 1})
            raise ValueError(f"Duplicate variables in ordering: {', '.join(dups)}")
        declared = set(ordering)
        missing = [v for v in needed if v not in declared]
        if missing:
            raise ValueError(
                "Ordering does not cover all basic events; missing: "
                + ", ".join(missing))
    bdd = BDD()
    bdd.declare(*ordering)
    return bdd


def exactly_k(bdd: BDD, funcs: List[Function], k: int) -> Function:
    return _exactly_k(bdd, funcs, k, {})


def _exactly_k(bdd: BDD, funcs: List[Function], k: int, memo: Dict[Tuple[int, int], Function]) -> Function:
    # The (len(funcs), k) memo key is sound only because every recursive call
    # sees a suffix of the same original list, so the length identifies it.
    # The memo must therefore never be shared across top-level calls.
    key = (len(funcs), k)
    if key in memo:
        return memo[key]
    if k < 0:
        r = bdd.false
    elif not funcs:
        r = bdd.true if k == 0 else bdd.false
    else:
        x, rest = funcs[0], funcs[1:]
        r = bdd.apply('|', bdd.apply('&', x, _exactly_k(bdd, rest, k - 1, memo)), bdd.apply('&', bdd.apply('~', x), _exactly_k(bdd, rest, k, memo)))
    memo[key] = r
    return r


def _at_least_k(bdd: BDD, funcs: List[Function], k: int, memo: Dict[Tuple[int, int], Function]) -> Function:
    # Same suffix-keyed memo invariant as _exactly_k.
    key = (len(funcs), k)
    if key in memo:
        return memo[key]
    if k <= 0:
        r = bdd.true
    elif not funcs:
        r = bdd.false
    else:
        x, rest = funcs[0], funcs[1:]
        r = bdd.apply('|', bdd.apply('&', x, _at_least_k(bdd, rest, k - 1, memo)), bdd.apply('&', bdd.apply('~', x), _at_least_k(bdd, rest, k, memo)))
    memo[key] = r
    return r


def at_least_k(bdd: BDD, funcs: List[Function], k: int) -> Function:
    return _at_least_k(bdd, funcs, k, {})


def _gate_semantics(node: Dict[str, Any], n_children: int, success_mode: bool) -> Tuple[str, Optional[int]]:
    """Single source of truth for gate meaning, including the success-mode dual.

    Returns (kind, k) with kind in {AND, OR, EXACTLY_K, NOT_EXACTLY_K, AT_LEAST_K}.
    Failure-mode gates are stated over failure indicators; in success mode the
    same tree is reinterpreted over success indicators y_i = 1 - x_i, so each
    gate maps to its De Morgan dual:
      AND <-> OR;  at-least-k -> at-least-(n-k+1);
      exactly-one(x) fails <-> NOT exactly-(n-1)(y)   [not NOT exactly-one(y),
      which is wrong for n >= 3].
    """
    gate = node.get("gate")
    if gate == "AND":
        return ("OR", None) if success_mode else ("AND", None)
    if gate == "OR":
        return ("AND", None) if success_mode else ("OR", None)
    if gate == "XOR":
        if success_mode:
            return ("NOT_EXACTLY_K", n_children - 1)
        return ("EXACTLY_K", 1)
    if gate == "K_OF_N":
        k = node.get("k")
        if not isinstance(k, int) or isinstance(k, bool) or k < 0:
            raise ValueError("K_OF_N gate requires integer field 'k' >= 0")
        if k > n_children:
            raise ValueError(
                f"K_OF_N gate has k={k} but only {n_children} children")
        if success_mode:
            return ("AT_LEAST_K", n_children - k + 1)
        return ("AT_LEAST_K", k)
    raise ValueError(f"Unsupported gate: {gate}")


def _walk(node: Dict[str, Any],
          registry: Optional[Dict[str, Dict[str, Any]]],
          success_mode: bool,
          leaf: Callable[[Dict[str, Any]], Any],
          combine: Callable[[str, Optional[int], List[Any]], Any],
          path: Optional[List[str]] = None) -> Any:
    """Shared traversal for all tree renderings (BDD / expression / symbolic).

    Resolves ref nodes through the registry and detects reference cycles by
    tracking the ids currently on the recursion path (re-visits off the path
    are legitimate DAG sharing, not cycles)."""
    if path is None:
        path = []
    if "ref" in node:
        if not registry:
            raise ValueError(f"Reference node '{node['ref']}' encountered without a node registry")
        target = registry.get(str(node["ref"]))
        if target is None:
            raise ValueError(f"Unknown ref: {node['ref']}")
        return _walk(target, registry, success_mode, leaf, combine, path)
    nid = str(node["id"]) if "id" in node else None
    if nid is not None and nid in path:
        cycle = path[path.index(nid):] + [nid]
        raise ValueError(
            "Cycle detected in tree (id revisited on the same path; a ref"
            " cycle or a duplicate id): " + " -> ".join(cycle))
    if nid is not None:
        path.append(nid)
    try:
        if _is_leaf(node):
            _leaf_id(node)
            return leaf(node)
        children = _children_of(node)
        if not children:
            raise ValueError(
                f"Gate '{node.get('gate')}' node '{nid}' has no children")
        parts = [_walk(ch, registry, success_mode, leaf, combine, path)
                 for ch in children]
        kind, k = _gate_semantics(node, len(parts), success_mode)
        return combine(kind, k, parts)
    finally:
        if nid is not None:
            path.pop()


def node_to_bdd(bdd: BDD, node: Dict[str, Any], registry: Optional[Dict[str, Dict[str, Any]]] = None, success_mode: bool = False) -> Function:
    def leaf(n: Dict[str, Any]) -> Function:
        return bdd.var(str(n["id"]))

    def combine(kind: str, k: Optional[int], parts: List[Function]) -> Function:
        if kind == "AND":
            r = bdd.true
            for f in parts:
                r = bdd.apply('&', r, f)
            return r
        if kind == "OR":
            r = bdd.false
            for f in parts:
                r = bdd.apply('|', r, f)
            return r
        if kind == "EXACTLY_K":
            return exactly_k(bdd, parts, k)
        if kind == "NOT_EXACTLY_K":
            return bdd.apply('~', exactly_k(bdd, parts, k))
        if kind == "AT_LEAST_K":
            return _at_least_k(bdd, parts, k, {})
        raise ValueError(f"Unsupported gate kind: {kind}")

    return _walk(node, registry, success_mode, leaf, combine)


def node_to_expr(node: Dict[str, Any], registry: Optional[Dict[str, Dict[str, Any]]] = None, use_names: bool = False, success_mode: bool = False) -> str:
    def leaf(n: Dict[str, Any]) -> str:
        if use_names:
            return str(n.get("name") or n.get("id"))
        return str(n.get("id"))

    def combine(kind: str, k: Optional[int], parts: List[str]) -> str:
        joined = ', '.join(parts)
        if kind == "AND":
            return f"AND({joined})"
        if kind == "OR":
            return f"OR({joined})"
        if kind == "EXACTLY_K":
            if k == 1:
                return f"EXACTLY_ONE({joined})"
            return f"EXACTLY_K({k}; {joined})"
        if kind == "NOT_EXACTLY_K":
            return f"NOT_EXACTLY_K({k}; {joined})"
        if kind == "AT_LEAST_K":
            return f"K_OF_N({k}; {joined})"
        raise ValueError(f"Unsupported gate kind: {kind}")

    return _walk(node, registry, success_mode, leaf, combine)


def node_to_symbolic(node: Dict[str, Any], registry: Optional[Dict[str, Dict[str, Any]]] = None, use_names: bool = False, success_mode: bool = False) -> str:
    def leaf(n: Dict[str, Any]) -> str:
        if use_names:
            return str(n.get("name") or n.get("id"))
        return str(n.get("id"))

    def combine(kind: str, k: Optional[int], parts: List[str]) -> str:
        joined = ', '.join(parts)
        if kind == "AND":
            if len(parts) == 1:
                return parts[0]
            return f"({' * '.join(parts)})"
        if kind == "OR":
            if len(parts) == 1:
                return parts[0]
            return f"({' + '.join(parts)})"
        if kind == "EXACTLY_K":
            if k == 1:
                return f"XOR({joined})"
            return f"EXACTLY_K({k}; {joined})"
        if kind == "NOT_EXACTLY_K":
            return f"NOT_EXACTLY_K({k}; {joined})"
        if kind == "AT_LEAST_K":
            return f"K_OF_N({k}; {joined})"
        raise ValueError(f"Unsupported gate kind: {kind}")

    return _walk(node, registry, success_mode, leaf, combine)


def gather_nodes(node: Dict[str, Any], acc: Dict[str, Dict[str, Any]]) -> None:
    if not isinstance(node, dict):
        raise ValueError(f"Tree node must be a JSON object, got: {node!r}")
    if "id" in node:
        acc[str(node["id"])] = node
    if "ref" in node:
        return
    for ch in _children_of(node):
        if "ref" not in ch:
            gather_nodes(ch, acc)


def var_probs_from_tree(tree: Dict[str, Any]) -> Dict[str, Any]:
    nodes: Dict[str, Dict[str, Any]] = {}
    gather_nodes(tree, nodes)
    probs: Dict[str, Any] = {}
    for nid, n in nodes.items():
        if _is_leaf(n):
            if "prob" in n and n["prob"] is not None:
                val = n["prob"]
                if isinstance(val, list):
                    val = np.array(val)
                probs[str(nid)] = val
    return probs


def load_probs_from_file(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
         raise ValueError(f"File not found: {path}")

    if path.endswith('.csv'):
        df = pd.read_csv(path)
    elif path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(path)
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel.")

    return {str(col): df[col].values for col in df.columns}


def align_probabilities(probs: Dict[str, Any], shuffle: bool = False, seed: Optional[int] = 0) -> Dict[str, Any]:
    """Align sample-array probabilities to a common length.

    Shorter arrays are bootstrap-resampled with replacement. This is a stopgap
    until a proper uncertainty engine (aligned joint sample matrix) exists:
    per-variable resampling destroys any correlation between events, so use
    equal-length, jointly drawn samples whenever correlations matter. The
    default fixed seed makes runs reproducible; pass seed=None for entropy."""
    rng = np.random.default_rng(seed)

    for k, v in probs.items():
        if isinstance(v, list):
            probs[k] = np.array(v)

    arrays = {k: v for k, v in probs.items() if isinstance(v, np.ndarray)}
    if not arrays:
        return probs

    max_len = max(len(v) for v in arrays.values())

    for k, v in arrays.items():
        if len(v) < max_len:
            probs[k] = rng.choice(v, max_len, replace=True)

        if shuffle:
             arr = probs[k].copy()
             rng.shuffle(arr)
             probs[k] = arr

    return probs


def wmc(bdd: BDD, func: Function, var_prob: Dict[str, Any], allow_missing: bool = False) -> Any:
    """Probability of `func` by memoized Shannon cofactor recursion, O(|BDD|):
    pr(f) = q * pr(f_high) + (1-q) * pr(f_low).

    Probabilities may be scalars or numpy sample arrays (evaluated elementwise
    via broadcasting). dd.autoref uses complemented edges: .high/.low always
    return the children of the un-negated node, so for a negated reference the
    complement is pushed into the children (~hi/~lo) and the memo is keyed on
    the signed reference. Computing "1 - r" at the reference instead would
    turn the engine's relative accuracy into ~1e-16 absolute and floor
    rare-event probabilities to 0; pushing the complement down keeps every
    partial result a non-negative sum of products."""
    memo: Dict[int, Any] = {}

    def rec(u: Function) -> Any:
        if u == bdd.true:
            return 1.0
        if u == bdd.false:
            return 0.0
        key = int(u)
        r = memo.get(key)
        if r is None:
            var = u.var
            pv = var_prob.get(var)
            if pv is None:
                if not allow_missing:
                    raise ValueError(
                        f"No probability defined for basic event '{var}'")
                pv = 0.0
            if isinstance(pv, (list, np.ndarray)):
                pv = np.asarray(pv, dtype=float)
            else:
                pv = float(pv)
            hi, lo = u.high, u.low
            if u.negated:
                hi, lo = ~hi, ~lo
            r = pv * rec(hi) + (1.0 - pv) * rec(lo)
            memo[key] = r
        return r

    return rec(func)


def normalize_tree(tree: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(tree, dict):
        raise ValueError(f"Tree must be a JSON object, got: {tree!r}")
    if "ft_nodes" in tree and "be_nodes" in tree:
        for kind in ("ft_nodes", "be_nodes"):
            for n in tree[kind]:
                if not isinstance(n, dict) or "label" not in n:
                    raise ValueError(
                        f"Flat-format {kind} entries must be objects with a"
                        f" 'label' field, got: {n!r}")
        ft_nodes = {n["label"]: n for n in tree["ft_nodes"]}
        be_nodes = {n["label"]: n for n in tree["be_nodes"]}
        all_nodes = {**ft_nodes, **be_nodes}

        root_id = None
        if "analysis" in tree and "esd_nodes" in tree:
            initiator = tree["analysis"].get("initiator")
            for esd in tree["esd_nodes"]:
                if esd["label"] == initiator:
                    root_id = esd.get("ft")
                    break

        if not root_id:
            referenced = set()
            for n in ft_nodes.values():
                for child in n.get("branches", []):
                    referenced.add(child)
            candidates = [n["label"] for n in ft_nodes.values() if n["label"] not in referenced]
            if candidates:
                root_id = candidates[0]
            elif ft_nodes:
                root_id = next(iter(ft_nodes))

        if not root_id:
             if be_nodes and len(be_nodes) == 1:
                 root_id = next(iter(be_nodes))
             else:
                raise ValueError("Could not determine root node in flat format")

        def build_recursive(nid: str, path: List[str]) -> Dict[str, Any]:
            if nid in path:
                cycle = path[path.index(nid):] + [nid]
                raise ValueError(
                    "Cycle detected in flat-format branches: " + " -> ".join(cycle))
            if nid not in all_nodes:
                 if not nid:
                     return None
                 raise ValueError(f"Node {nid} not found in definitions")

            node_def = all_nodes[nid]
            new_node = {
                "id": nid,
                "name": nid,
            }

            if nid in be_nodes:
                new_node["event_type"] = "basic"
                new_node["gate"] = None
                new_node["prob"] = node_def.get("prob")
                new_node["children"] = []
            else:
                new_node["event_type"] = "top" if nid == root_id else "intermediate"
                new_node["gate"] = node_def.get("gate")
                if "k" in node_def:
                    new_node["k"] = node_def.get("k")
                children = []
                path.append(nid)
                try:
                    for child_id in node_def.get("branches", []):
                        child = build_recursive(child_id, path)
                        if child:
                            children.append(child)
                finally:
                    path.pop()
                new_node["children"] = children
            return new_node

        return build_recursive(root_id, [])
    return tree


def _validate_probs(tree: Dict[str, Any], probs: Dict[str, Any], allow_missing: bool) -> None:
    vars_out: List[str] = []
    collect_basic_events(tree, vars_out)
    needed = list(dict.fromkeys(vars_out))
    missing = [v for v in needed if v not in probs]
    if missing and not allow_missing:
        raise ValueError(
            "No probability defined for basic events: " + ", ".join(missing)
            + ". Provide 'prob' in the tree, a probability file, or pass"
            " allow_missing=True (--assume-missing-zero) to treat them as 0.")
    non_numeric = []
    non_finite = []
    out_of_range = []
    for v in needed:
        if v not in probs:
            continue
        try:
            arr = np.asarray(probs[v], dtype=float)
        except (ValueError, TypeError):
            non_numeric.append(v)
            continue
        if not np.all(np.isfinite(arr)):
            non_finite.append(v)
        elif np.any(arr < 0.0) or np.any(arr > 1.0):
            out_of_range.append(v)
    problems = []
    if non_numeric:
        problems.append("non-numeric values for: " + ", ".join(non_numeric))
    if non_finite:
        problems.append(
            "non-finite (NaN/inf) values for: " + ", ".join(non_finite)
            + " (note: pandas pads ragged CSV/Excel columns with NaN —"
            " columns must have equal length)")
    if out_of_range:
        problems.append("values outside [0, 1] for: " + ", ".join(out_of_range))
    if problems:
        raise ValueError("Invalid probabilities: " + "; ".join(problems))


def compute_event_probabilities(bdd: BDD, tree: Dict[str, Any], probs_by_id: Union[Dict[str, Any], str, None] = None, success_mode: bool = False, shuffle: bool = False, seed: Optional[int] = 0, allow_missing: bool = False) -> Dict[str, Any]:
    tree = normalize_tree(tree)
    base_probs = var_probs_from_tree(tree)

    if probs_by_id:
        if isinstance(probs_by_id, str):
            file_probs = load_probs_from_file(probs_by_id)
            base_probs.update(file_probs)
        elif isinstance(probs_by_id, dict):
             base_probs.update(probs_by_id)

    base_probs = align_probabilities(base_probs, shuffle=shuffle, seed=seed)
    _validate_probs(tree, base_probs, allow_missing)

    nodes: Dict[str, Dict[str, Any]] = {}
    gather_nodes(tree, nodes)
    prob_out: Dict[str, Any] = {}
    bdd_cache: Dict[str, Function] = {}
    def build_cached(n: Dict[str, Any]) -> Function:
        nid = str(n["id"])
        if nid in bdd_cache:
            return bdd_cache[nid]
        f = node_to_bdd(bdd, n, nodes, success_mode=success_mode)
        bdd_cache[nid] = f
        return f
    for nid, n in nodes.items():
        f = build_cached(n)
        prob_out[nid] = wmc(bdd, f, base_probs, allow_missing=allow_missing)
    return prob_out


def build(tree: Dict[str, Any], ordering: Optional[List[str]] = None, use_names: bool = False, success_mode: bool = False) -> Tuple[BDD, Function, str, str]:
    tree = normalize_tree(tree)
    bdd = build_manager(tree, ordering)
    nodes: Dict[str, Dict[str, Any]] = {}
    gather_nodes(tree, nodes)
    top_func = node_to_bdd(bdd, tree, nodes, success_mode=success_mode)
    expr = node_to_expr(tree, nodes, use_names=use_names, success_mode=success_mode)
    symb = node_to_symbolic(tree, nodes, use_names=use_names, success_mode=success_mode)
    return bdd, top_func, expr, symb
