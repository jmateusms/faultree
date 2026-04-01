from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import os

try:
    from dd.autoref import BDD, Function
except Exception as e:
    raise RuntimeError("dd package is required. Install with `pip install dd`.") from e


def collect_basic_events(node: Dict[str, Any], vars_out: List[str]) -> None:
    if "ref" in node:
        return
    et = node.get("event_type")
    gate = node.get("gate")
    children = node.get("children", [])
    if not children and (et in ("basic", "undeveloped") or gate in (None, "BASIC")):
        vars_out.append(str(node["id"]))
        return
    for ch in children:
        collect_basic_events(ch, vars_out)


def build_manager(tree: Dict[str, Any], ordering: Optional[List[str]] = None) -> BDD:
    vars_out: List[str] = []
    collect_basic_events(tree, vars_out)
    if ordering is None:
        ordering = list(dict.fromkeys(vars_out))
    bdd = BDD()
    bdd.declare(*ordering)
    return bdd


def exactly_k(bdd: BDD, funcs: List[Function], k: int, memo: Dict[Tuple[int, int], Function]) -> Function:
    key = (len(funcs), k)
    if key in memo:
        return memo[key]
    if k < 0:
        r = bdd.false
    elif not funcs:
        r = bdd.true if k == 0 else bdd.false
    else:
        x, rest = funcs[0], funcs[1:]
        r = bdd.apply('|', bdd.apply('&', x, exactly_k(bdd, rest, k - 1, memo)), bdd.apply('&', bdd.apply('~', x), exactly_k(bdd, rest, k, memo)))
    memo[key] = r
    return r


def _at_least_k(bdd: BDD, funcs: List[Function], k: int, memo: Dict[Tuple[int, int], Function]) -> Function:
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
    memo: Dict[Tuple[int, int], Function] = {}
    return exactly_k(bdd, funcs, k, memo) if k == len(funcs) else _at_least_k(bdd, funcs, k, memo)


def node_to_bdd(bdd: BDD, node: Dict[str, Any], registry: Optional[Dict[str, Dict[str, Any]]] = None, success_mode: bool = False) -> Function:
    if registry and "ref" in node:
        target = registry.get(str(node["ref"]))
        if target is None:
            raise ValueError(f"Unknown ref: {node['ref']}")
        return node_to_bdd(bdd, target, registry, success_mode)
    gate = node.get("gate")
    children = node.get("children", [])
    if not children and gate in (None, "BASIC"):
        return bdd.var(str(node["id"]))
    child_funcs = [node_to_bdd(bdd, ch, registry, success_mode) for ch in children]
    
    target_gate = gate
    if success_mode:
        if gate == "AND": target_gate = "OR"
        elif gate == "OR": target_gate = "AND"
    
    if target_gate == "AND":
        r = bdd.true
        for f in child_funcs:
            r = bdd.apply('&', r, f)
        return r
    if target_gate == "OR":
        r = bdd.false
        for f in child_funcs:
            r = bdd.apply('|', r, f)
        return r
    if gate == "XOR":
        res = exactly_k(bdd, child_funcs, 1, {})
        if success_mode:
            return bdd.apply('~', res)
        return res
    if gate == "K_OF_N":
        k = node.get("k")
        if not isinstance(k, int) or k < 0:
            raise ValueError("K_OF_N gate requires integer field 'k' >= 0")
        if success_mode:
            n = len(child_funcs)
            k_dual = n - k + 1
            return at_least_k(bdd, child_funcs, k_dual)
        return at_least_k(bdd, child_funcs, k)
    raise ValueError(f"Unsupported gate: {gate}")


def node_to_expr(node: Dict[str, Any], registry: Optional[Dict[str, Dict[str, Any]]] = None, use_names: bool = False, success_mode: bool = False) -> str:
    if registry and "ref" in node:
        target = registry.get(str(node["ref"]))
        if target is None:
            raise ValueError(f"Unknown ref: {node['ref']}")
        return node_to_expr(target, registry, use_names, success_mode)
    gate = node.get("gate")
    children = node.get("children", [])
    if use_names:
        name = str(node.get("name") or node.get("id"))
    else:
        name = str(node.get("id"))
    if not children and gate in (None, "BASIC"):
        return name
    parts = [node_to_expr(ch, registry, use_names, success_mode) for ch in children]
    
    target_gate = gate
    if success_mode:
        if gate == "AND": target_gate = "OR"
        elif gate == "OR": target_gate = "AND"
        
    if target_gate == "AND":
        return f"AND({', '.join(parts)})"
    if target_gate == "OR":
        return f"OR({', '.join(parts)})"
    if gate == "XOR":
        if success_mode:
             return f"NOT_EXACTLY_ONE({', '.join(parts)})"
        return f"EXACTLY_ONE({', '.join(parts)})"
    if gate == "K_OF_N":
        k = node.get("k")
        if success_mode:
            n = len(parts)
            k = n - k + 1
        return f"K_OF_N({k}; {', '.join(parts)})"
    raise ValueError(f"Unsupported gate: {gate}")


def node_to_symbolic(node: Dict[str, Any], registry: Optional[Dict[str, Dict[str, Any]]] = None, use_names: bool = False, success_mode: bool = False) -> str:
    if registry and "ref" in node:
        target = registry.get(str(node["ref"]))
        if target is None:
            raise ValueError(f"Unknown ref: {node['ref']}")
        return node_to_symbolic(target, registry, use_names, success_mode)
    gate = node.get("gate")
    children = node.get("children", [])
    if use_names:
        name = str(node.get("name") or node.get("id"))
    else:
        name = str(node.get("id"))
    if not children and gate in (None, "BASIC"):
        return name
    parts = [node_to_symbolic(ch, registry, use_names, success_mode) for ch in children]
    
    target_gate = gate
    if success_mode:
        if gate == "AND": target_gate = "OR"
        elif gate == "OR": target_gate = "AND"
        
    if target_gate == "AND":
        if len(parts) == 1: return parts[0]
        return f"({' * '.join(parts)})"
    if target_gate == "OR":
        if len(parts) == 1: return parts[0]
        return f"({' + '.join(parts)})"
    if gate == "XOR":
        if success_mode:
            return f"XNOR({', '.join(parts)})"
        return f"XOR({', '.join(parts)})"
    if gate == "K_OF_N":
        k = node.get("k")
        if success_mode:
            n = len(parts)
            k = n - k + 1
        return f"K_OF_N({k}; {', '.join(parts)})"
    raise ValueError(f"Unsupported gate: {gate}")


def gather_nodes(node: Dict[str, Any], acc: Dict[str, Dict[str, Any]]) -> None:
    if "id" in node:
        acc[str(node["id"])] = node
    for ch in node.get("children", []):
        if "ref" not in ch:
            gather_nodes(ch, acc)


def var_probs_from_tree(tree: Dict[str, Any]) -> Dict[str, Any]:
    nodes: Dict[str, Dict[str, Any]] = {}
    gather_nodes(tree, nodes)
    probs: Dict[str, Any] = {}
    for nid, n in nodes.items():
        et = n.get("event_type")
        gate = n.get("gate")
        children = n.get("children", [])
        if not children and (et in ("basic", "undeveloped") or gate in (None, "BASIC")):
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


def align_probabilities(probs: Dict[str, Any], shuffle: bool = False) -> Dict[str, Any]:
    # Ensure lists are numpy arrays
    for k, v in probs.items():
        if isinstance(v, list):
            probs[k] = np.array(v)
            
    # Identify arrays
    arrays = {k: v for k, v in probs.items() if isinstance(v, np.ndarray)}
    if not arrays:
        return probs
        
    # Find max size
    max_len = max(len(v) for v in arrays.values())
    
    # Resample smaller arrays
    for k, v in arrays.items():
        if len(v) < max_len:
            # Sample with replacement
            probs[k] = np.random.choice(v, max_len, replace=True)
        
        if shuffle:
             # Shuffle the array
             # We create a copy to avoid mutating original if referenced elsewhere
             arr = probs[k].copy()
             np.random.shuffle(arr)
             probs[k] = arr
             
    return probs


def wmc(bdd: BDD, func: Function, var_prob: Dict[str, Any]) -> Any:
    total = 0.0
    for assign in bdd.pick_iter(func):
        p = 1.0
        for var, val in assign.items():
            pv = var_prob.get(var, 0.0)
            if isinstance(pv, (int, float)):
                pv = float(pv)
            
            if bool(val):
                p = p * pv
            else:
                p = p * (1.0 - pv)
        total = total + p
    return total


def normalize_tree(tree: Dict[str, Any]) -> Dict[str, Any]:
    if "ft_nodes" in tree and "be_nodes" in tree:
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

        def build_recursive(nid: str) -> Dict[str, Any]:
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
                for child_id in node_def.get("branches", []):
                    child = build_recursive(child_id)
                    if child:
                        children.append(child)
                new_node["children"] = children
            return new_node

        return build_recursive(root_id)
    return tree


def compute_event_probabilities(bdd: BDD, tree: Dict[str, Any], probs_by_id: Union[Dict[str, Any], str, None] = None, success_mode: bool = False, shuffle: bool = False) -> Dict[str, Any]:
    tree = normalize_tree(tree)
    base_probs = var_probs_from_tree(tree)
    
    if probs_by_id:
        if isinstance(probs_by_id, str):
            # Load from file
            file_probs = load_probs_from_file(probs_by_id)
            base_probs.update(file_probs)
        elif isinstance(probs_by_id, dict):
             # Handle possible file reference in dict if needed, but for now assume direct mapping
             # Check if any value is a file string?
             # "allow the user to provide a reference to a xlsx or csv file with the event id as column name"
             # This is handled by passing string to probs_by_id
             base_probs.update(probs_by_id)
    
    # Align probabilities (handle arrays, resizing, shuffling)
    base_probs = align_probabilities(base_probs, shuffle=shuffle)
    
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
        prob_out[nid] = wmc(bdd, f, base_probs)
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
