from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import csv
import io
import math
import numpy as np
import os
import re

try:
    from dd.autoref import BDD, Function
except Exception as e:
    raise RuntimeError("dd package is required. Install with `pip install dd`.") from e


@dataclass(frozen=True)
class CompiledTree:
    """A reusable representation of one normalized fault tree.

    Field bindings are frozen, but the BDD manager and mappings are mutable;
    callers must not modify them. The source tree is copied on compilation.

    The BDD functions for the top event and every named event are built once.
    Reuse this object when quantifying the same structure with several
    probability maps; it avoids rebuilding the manager and traversing the
    tree for each scenario.
    """

    tree: Dict[str, Any]
    bdd: BDD
    top: Function
    expression: str
    symbolic: str
    event_functions: Dict[str, Function]
    ordering: Tuple[str, ...]
    success_mode: bool


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


def _node_signature(value: Any) -> Any:
    """Return a comparison-safe, deterministic representation of JSON input."""
    if isinstance(value, dict):
        return tuple(sorted((str(k), _node_signature(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_node_signature(v) for v in value)
    if isinstance(value, np.ndarray):
        return ("ndarray", tuple(_node_signature(v) for v in value.tolist()))
    if isinstance(value, np.generic):
        return value.item()
    return value


def _add_node_definition(acc: Dict[str, Dict[str, Any]], nid: str,
                         node: Dict[str, Any]) -> None:
    """Register an event definition, permitting only exact repeated copies.

    Repeating an identical definition is useful for formats that inline a
    shared event.  A repeated id with different content is ambiguous and must
    never silently replace the definition used by refs or result reporting.
    """
    previous = acc.get(nid)
    if previous is None:
        acc[nid] = node
    elif _node_signature(previous) != _node_signature(node):
        raise ValueError(
            f"Conflicting duplicate id '{nid}' (event id): repeated definitions "
            "must be identical (or use {'ref': id} for sharing)")


def gather_nodes(node: Dict[str, Any], acc: Dict[str, Dict[str, Any]]) -> None:
    positions: Dict[str, str] = {}

    def visit(current: Dict[str, Any], path: str) -> None:
        if not isinstance(current, dict):
            raise ValueError(f"Tree node at {path} must be a JSON object")
        if "ref" in current:
            return
        if "id" in current:
            nid = str(current["id"])
            try:
                _add_node_definition(acc, nid, current)
            except ValueError as error:
                raise ValueError(f"{error}; positions: {positions.get(nid, 'existing registry')} and {path}") from error
            positions.setdefault(nid, path)
        for index, child in enumerate(_children_of(current)):
            visit(child, f"{path}.children[{index}]")

    visit(node, "root")


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


_DECIMAL_COMMA = re.compile(r"^[+-]?(\d+,\d*|,\d+)([eE][+-]?\d+)?$")


def _csv_delimiter(text: str) -> str:
    """Column separator of a probability CSV, from its header line.

    Spreadsheets in locales with a decimal comma (pt-BR Excel, for one)
    write ';' between columns; tabs come from copy-and-paste. A header with
    no separator at all is a single column, where a comma can only be a
    decimal comma."""
    header = next((line for line in text.splitlines() if line.strip()), "")
    counts = {sep: header.count(sep) for sep in (",", ";", "\t")}
    best = max(counts, key=lambda sep: (counts[sep], sep == ","))
    return best if counts[best] else ";"


def _read_csv_columns(path: str) -> Dict[str, np.ndarray]:
    """Read a header-plus-rows CSV into float columns without pandas.

    The separator is ',', ';' or a tab (see ``_csv_delimiter``); with ';' or
    a tab, decimal commas such as ``0,05`` are read as ``0.05``. Empty cells
    (e.g. a ragged column) become NaN so validation reports them, exactly as
    the previous pandas-based reader did. Columns without a header are
    ignored when they are empty too (trailing separators, as some
    spreadsheet exports write); an unnamed column holding values is an error."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        text = fh.read()
    delimiter = _csv_delimiter(text)
    rows = [row for row in csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
            if any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError(f"Probability file is empty: {path}")
    header = [cell.strip() for cell in rows[0]]
    named = [name for name in header if name]
    duplicated = sorted({name for name in named if named.count(name) > 1})
    if duplicated:
        raise ValueError(f"{path}: column names must be unique"
                         f" (repeated: {', '.join(duplicated)})")
    columns: Dict[str, List[float]] = {name: [] for name in named}
    for line_no, row in enumerate(rows[1:], start=2):
        if len(row) > len(header):
            hint = (" (with decimal commas, separate the columns with ';')"
                    if delimiter == "," else "")
            raise ValueError(f"{path}:{line_no}: more cells than header columns" + hint)
        for index, cell in enumerate(row + [""] * (len(header) - len(row))):
            name, cell = header[index], cell.strip()
            if not name:
                if cell:
                    raise ValueError(
                        f"{path}:{line_no}: value {cell!r} in column {index + 1},"
                        " which has no name in the header row")
                continue
            if delimiter != "," and _DECIMAL_COMMA.match(cell):
                cell = cell.replace(",", ".")
            try:
                columns[name].append(float(cell) if cell else math.nan)
            except ValueError:
                raise ValueError(
                    f"{path}:{line_no}: non-numeric probability {cell!r} in column {name!r}") from None
    return {name: np.asarray(values, dtype=float) for name, values in columns.items()}


def load_probs_from_file(path: str) -> Dict[str, np.ndarray]:
    """Load one probability column per basic-event id from CSV or Excel.

    CSV needs only the standard library. Excel needs the optional
    ``faultree[excel]`` extra (pandas + openpyxl), imported on demand."""
    if not os.path.exists(path):
         raise ValueError(f"File not found: {path}")

    extension = os.path.splitext(path)[1].lower()
    if extension == '.csv':
        return _read_csv_columns(path)
    if extension in ('.xls', '.xlsx'):
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "Reading Excel probability files needs pandas and openpyxl: "
                'pip install "faultree[excel]" (or use a CSV file).') from e
        df = pd.read_excel(path)
        return {str(col): df[col].values for col in df.columns}
    raise ValueError("Unsupported file format. Use CSV or Excel.")


def align_probabilities(probs: Dict[str, Any], shuffle: bool = False,
                        seed: Optional[int] = 0,
                        resample_independent: bool = False) -> Dict[str, Any]:
    """Align sample-array probabilities to a common length.

    Equal-length vectors are interpreted as aligned joint samples.  Different
    lengths are rejected by default because independently resampling columns
    would destroy that alignment.  Set ``resample_independent=True`` only to
    opt into independent marginal bootstrap resampling.  ``shuffle`` applies
    one shared permutation to all aligned vectors, preserving joint samples.
    The default seed makes the explicit stochastic operation reproducible."""
    rng = np.random.default_rng(seed)
    probs = dict(probs)
    for k, v in probs.items():
        if isinstance(v, (list, tuple)):
            probs[k] = np.asarray(v)

    arrays = {k: v for k, v in probs.items()
              if isinstance(v, np.ndarray) and v.ndim == 1}
    if not arrays:
        return probs

    lengths = {len(v) for v in arrays.values()}
    if len(lengths) != 1 and not resample_independent:
        details = ", ".join(f"{key}={len(value)}" for key, value in arrays.items())
        raise ValueError(
            "Sample probability vectors must have equal lengths to preserve "
            "joint-sample alignment; got " + details + ". Pass "
            "resample_independent=True (--resample-independent) to explicitly "
            "bootstrap independent marginals.")
    max_len = max(lengths)

    for k, v in arrays.items():
        if resample_independent:
            probs[k] = rng.choice(v, max_len, replace=True)

    if shuffle:
        permutation = rng.permutation(max_len)
        for k, v in arrays.items():
            probs[k] = probs[k][permutation]

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
            if not isinstance(tree[kind], list):
                raise ValueError(f"Flat-format '{kind}' must be a list")
            for n in tree[kind]:
                if (not isinstance(n, dict) or not isinstance(n.get("label"), str)
                        or not n["label"]):
                    raise ValueError(
                        f"Flat-format {kind} entries must be objects with a"
                        f" non-empty string 'label' field, got: {n!r}")

        def indexed(kind: str) -> Dict[str, Dict[str, Any]]:
            result: Dict[str, Dict[str, Any]] = {}
            positions = {}
            for index, definition in enumerate(tree[kind]):
                label = definition["label"]
                previous = result.get(label)
                if previous is None:
                    result[label] = definition
                    positions[label] = index
                elif _node_signature(previous) != _node_signature(definition):
                    raise ValueError(
                        f"Conflicting duplicate id '{label}' (event id) in {kind}: "
                        f"positions {kind}[{positions[label]}] and {kind}[{index}]; "
                        "repeated flat definitions must be identical")
            return result

        ft_nodes = indexed("ft_nodes")
        be_nodes = indexed("be_nodes")
        overlap = sorted(set(ft_nodes) & set(be_nodes))
        if overlap:
            raise ValueError("Conflicting duplicate event ids across ft_nodes "
                             "and be_nodes: " + ", ".join(overlap))
        all_nodes = {**ft_nodes, **be_nodes}

        for label, definition in ft_nodes.items():
            branches = definition.get("branches", [])
            if not isinstance(branches, list) or not all(isinstance(x, str) and x for x in branches):
                raise ValueError(
                    f"Flat-format branches for '{label}' must be a list of "
                    "non-empty event ids")

        # Diagnose graph cycles before inferring a root: a closed cycle has
        # no unreferenced node, but the cycle is the actionable error.
        def check_flat_cycles(nid: str, path: List[str]) -> None:
            if nid in path:
                cycle = path[path.index(nid):] + [nid]
                raise ValueError(
                    "Cycle detected in flat-format branches: " + " -> ".join(cycle))
            if nid not in ft_nodes:
                return
            for child_id in ft_nodes[nid].get("branches", []):
                check_flat_cycles(child_id, path + [nid])

        for label in ft_nodes:
            check_flat_cycles(label, [])

        root_id = None
        if "analysis" in tree and "esd_nodes" in tree:
            if not isinstance(tree["analysis"], dict) or not isinstance(tree["esd_nodes"], list):
                raise ValueError("Flat-format analysis/esd_nodes must be an object and a list")
            initiator = tree["analysis"].get("initiator")
            for esd in tree["esd_nodes"]:
                if isinstance(esd, dict) and esd.get("label") == initiator:
                    root_id = esd.get("ft")
                    break
            if initiator is not None and not root_id:
                raise ValueError(
                    f"Flat-format analysis initiator '{initiator}' has no explicit root")

        if not root_id:
            referenced = set()
            for n in ft_nodes.values():
                for child in n.get("branches", []):
                    referenced.add(child)
            candidates = sorted(set(all_nodes) - referenced)
            if len(candidates) != 1:
                raise ValueError(
                    "Flat format requires an explicit analysis/esd root or "
                    "exactly one unreferenced root; candidates: "
                    + ", ".join(candidates or ["none"]))
            root_id = candidates[0]
        if root_id not in all_nodes:
            raise ValueError(f"Flat-format root '{root_id}' not found in definitions")

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

        normalized = build_recursive(root_id, [])
        reachable: Dict[str, Dict[str, Any]] = {}
        gather_nodes(normalized, reachable)
        unreachable = sorted(set(all_nodes) - set(reachable))
        if unreachable:
            raise ValueError("Flat-format contains disconnected/inaccessible "
                             "nodes: " + ", ".join(unreachable))
        return normalized
    # Also validate recursive trees early, before BDD construction or output
    # maps can silently overwrite an id.
    registry: Dict[str, Dict[str, Any]] = {}
    gather_nodes(tree, registry)
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
    invalid_shape = []
    for v in needed:
        if v not in probs:
            continue
        value = probs[v]
        try:
            raw = np.asarray(value)
            # JSON/API callers must supply actual numeric scalars/vectors;
            # accepting strings such as "0.2" hides malformed input.
            if raw.dtype.kind not in "iuf":
                raise TypeError("probability is not numeric")
            arr = raw.astype(float)
        except (ValueError, TypeError):
            non_numeric.append(v)
            continue
        if arr.ndim > 1 or (arr.ndim == 1 and arr.size == 0):
            invalid_shape.append(v)
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
            + " (note: ragged CSV/Excel columns are padded with NaN —"
            " columns must have equal length)")
    if invalid_shape:
        problems.append("probabilities must be finite scalars or non-empty "
                        "1-D vectors for: " + ", ".join(invalid_shape))
    if out_of_range:
        problems.append("values outside [0, 1] for: " + ", ".join(out_of_range))
    if problems:
        raise ValueError("Invalid probabilities: " + "; ".join(problems))


def _prepared_probabilities(tree: Dict[str, Any], probs_by_id: Union[Dict[str, Any], str, None],
                            *, shuffle: bool, seed: Optional[int],
                            allow_missing: bool,
                            resample_independent: bool) -> Dict[str, Any]:
    """Resolve and validate input probabilities once for a quantification."""
    base_probs = var_probs_from_tree(tree)
    if probs_by_id:
        if isinstance(probs_by_id, str):
            base_probs.update(load_probs_from_file(probs_by_id))
        elif isinstance(probs_by_id, dict):
            base_probs.update(probs_by_id)
        else:
            raise TypeError("probabilities must be a mapping, file path, or None")
    _validate_probs(tree, base_probs, allow_missing)
    base_probs = align_probabilities(
        base_probs, shuffle=shuffle, seed=seed,
        resample_independent=resample_independent)
    _validate_probs(tree, base_probs, allow_missing)
    return base_probs


def _faultree_version() -> str:
    try:
        return version("faultree")
    except PackageNotFoundError:
        # Useful for source checkouts that have not been installed yet.
        return "0.3.0"


def compile_tree(tree: Dict[str, Any], ordering: Optional[List[str]] = None,
                 use_names: bool = False, success_mode: bool = False,
                 *, include_expressions: bool = True) -> CompiledTree:
    """Compile a tree once for repeated quantification or inspection.

    ``ordering`` remains entirely caller-selected. This function deliberately
    does not apply a heuristic or mutate the declared model order.
    """
    normalized = deepcopy(normalize_tree(tree))
    bdd = build_manager(normalized, ordering)
    nodes: Dict[str, Dict[str, Any]] = {}
    gather_nodes(normalized, nodes)
    event_functions = {
        nid: node_to_bdd(bdd, node, nodes, success_mode=success_mode)
        for nid, node in nodes.items()
    }
    top_id = str(normalized["id"])
    return CompiledTree(
        tree=normalized,
        bdd=bdd,
        top=event_functions[top_id],
        expression=node_to_expr(normalized, nodes, use_names=use_names,
                                success_mode=success_mode) if include_expressions else "",
        symbolic=node_to_symbolic(normalized, nodes, use_names=use_names,
                                  success_mode=success_mode) if include_expressions else "",
        event_functions=event_functions,
        ordering=tuple(sorted(bdd.vars, key=bdd.vars.get)),
        success_mode=success_mode,
    )


def quantify_compiled(compiled: CompiledTree,
                      probs_by_id: Union[Dict[str, Any], str, None] = None,
                      *, shuffle: bool = False, seed: Optional[int] = 0,
                      allow_missing: bool = False,
                      resample_independent: bool = False) -> Dict[str, Any]:
    """Evaluate all event functions of a previously :func:`compile_tree` call."""
    base_probs = _prepared_probabilities(
        compiled.tree, probs_by_id, shuffle=shuffle, seed=seed,
        allow_missing=allow_missing, resample_independent=resample_independent)
    return {
        nid: wmc(compiled.bdd, function, base_probs, allow_missing=allow_missing)
        for nid, function in compiled.event_functions.items()
    }


def to_jsonable(value: Any) -> Any:
    """Convert results to strict-JSON types: numpy scalars/arrays become
    native numbers/lists and non-finite floats (an ``inf`` RRW, an undefined
    0/0 ratio) become ``None``, i.e. JSON ``null``."""
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def _ratio(numerator: Any, denominator: Any) -> Any:
    """Elementwise numerator / denominator: x/0 -> inf, 0/0 -> NaN, no warnings."""
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.true_divide(numerator, denominator)
    return float(value) if np.ndim(value) == 0 else value


def analyze(tree: Dict[str, Any], probs_by_id: Union[Dict[str, Any], str, None] = None,
            ordering: Optional[List[str]] = None, *, use_names: bool = False,
            success_mode: bool = False, shuffle: bool = False,
            seed: Optional[int] = 0, allow_missing: bool = False,
            resample_independent: bool = False) -> Dict[str, Any]:
    """Return a reproducible, structured FTA result.

    ``conditional_Q[event]["true"]`` and ``["false"]`` are top-event
    probabilities after forcing that *basic* event respectively true or false.
    Birnbaum importance is their signed difference. Its sign is retained: XOR
    and other non-monotonic logic can make forcing a failure less likely to
    produce the top event.

    ``importance[event]`` adds the usual failure-space measures, all exact
    (computed on the BDD, no cut-set approximation), with ``Q1``/``Q0`` the
    conditional probabilities above and ``q`` the event probability:

    - ``birnbaum`` = Q1 - Q0;
    - ``criticality`` = birnbaum * q / Q = (Q - Q0) / Q = 1 - 1/RRW, the
      (failure-oriented) criticality importance of Rausand & Hoyland. PRA
      codes often report this risk-decrease ratio as "Fussell-Vesely", but
      it is not Fussell's cut-set definition, P(some minimal cut set
      containing the event has failed | top event): for coherent trees that
      value is >= criticality and the two agree only when cut-set
      probabilities are small (rare-event approximation);
    - ``raw`` (risk achievement worth) = Q1 / Q;
    - ``rrw`` (risk reduction worth) = Q / Q0 (``inf`` when Q0 = 0).

    Undefined ratios (0/0) are NaN. For non-coherent logic (XOR) the same
    formulas hold, but criticality can be negative and RAW or RRW below 1,
    and the probabilistic readings of the coherent case no longer apply. In
    ``success_mode`` the result is a reliability, for which these
    failure-space ratios are not defined, so ``importance`` is ``None``.
    """
    compiled = compile_tree(tree, ordering, use_names=use_names,
                            success_mode=success_mode)
    base_probs = _prepared_probabilities(
        compiled.tree, probs_by_id, shuffle=shuffle, seed=seed,
        allow_missing=allow_missing, resample_independent=resample_independent)
    probabilities = {
        nid: wmc(compiled.bdd, function, base_probs, allow_missing=allow_missing)
        for nid, function in compiled.event_functions.items()
    }
    basic_events: List[str] = []
    collect_basic_events(compiled.tree, basic_events)
    conditional_q: Dict[str, Dict[str, Any]] = {}
    birnbaum: Dict[str, Any] = {}
    for event_id in dict.fromkeys(basic_events):
        forced_true = compiled.bdd.let({event_id: True}, compiled.top)
        forced_false = compiled.bdd.let({event_id: False}, compiled.top)
        q_true = wmc(compiled.bdd, forced_true, base_probs,
                     allow_missing=allow_missing)
        q_false = wmc(compiled.bdd, forced_false, base_probs,
                      allow_missing=allow_missing)
        conditional_q[event_id] = {"true": q_true, "false": q_false}
        birnbaum[event_id] = q_true - q_false
    top_id = str(compiled.tree["id"])
    importance = None
    if not success_mode:
        top_q = probabilities[top_id]
        importance = {
            event_id: {
                "birnbaum": birnbaum[event_id],
                "criticality": _ratio(birnbaum[event_id] * base_probs.get(event_id, 0.0), top_q),
                "raw": _ratio(conditional_q[event_id]["true"], top_q),
                "rrw": _ratio(top_q, conditional_q[event_id]["false"]),
            }
            for event_id in birnbaum
        }
    return {
        "Q": probabilities[top_id],
        "conditional_Q": conditional_q,
        "birnbaum": birnbaum,
        "importance": importance,
        "assumptions": {
            "basic_event_independence": True,
            "variable_order": list(compiled.ordering),
            "faultree_version": _faultree_version(),
            "seed": seed,
            "shuffle": shuffle,
            "resample_independent": resample_independent,
            "sample_vectors": "jointly_aligned unless resample_independent is true",
            "sample_count": max((np.asarray(v).size for v in base_probs.values()), default=1),
            "success_mode": success_mode,
        },
        "probabilities": probabilities,
        "expression": compiled.expression,
        "symbolic": compiled.symbolic,
    }


def benchmark_orderings(tree: Dict[str, Any], orderings: Iterable[List[str]],
                        *, success_mode: bool = False,
                        probs_by_id: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Measure caller-supplied BDD orders without choosing one on their behalf."""
    results: List[Dict[str, Any]] = []
    for ordering in orderings:
        started = perf_counter()
        compiled = compile_tree(tree, list(ordering), success_mode=success_mode,
                                include_expressions=False)
        compile_seconds = perf_counter() - started
        measurement = {}
        if probs_by_id is not None:
            evaluation_started = perf_counter()
            q = quantify_compiled(compiled, probs_by_id)[str(compiled.tree["id"])]
            measurement = {"evaluation_seconds": perf_counter() - evaluation_started,
                           "sample_count": int(np.asarray(q).size),
                           "Q": np.asarray(q).tolist()}
        results.append({
            "ordering": list(compiled.ordering),
            "bdd_nodes": len(compiled.bdd),
            "top_bdd_nodes": compiled.top.dag_size,
            "compile_seconds": compile_seconds,
            **measurement,
        })
    return results


def compute_event_probabilities(bdd: BDD, tree: Dict[str, Any], probs_by_id: Union[Dict[str, Any], str, None] = None, success_mode: bool = False, shuffle: bool = False, seed: Optional[int] = 0, allow_missing: bool = False, resample_independent: bool = False) -> Dict[str, Any]:
    tree = normalize_tree(tree)
    base_probs = _prepared_probabilities(
        tree, probs_by_id, shuffle=shuffle, seed=seed,
        allow_missing=allow_missing, resample_independent=resample_independent)

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
    compiled = compile_tree(tree, ordering, use_names=use_names,
                            success_mode=success_mode)
    return compiled.bdd, compiled.top, compiled.expression, compiled.symbolic
