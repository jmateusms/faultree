"""Reliability block diagrams (RBD), converted to and from fault trees.

An RBD describes when a system works: a block works when its component does,
and the system works when the diagram has a path of working blocks from its
input to its output. A fault tree describes when the same system fails, so
the two are De Morgan duals of each other:

- a ``series`` group fails when any block fails: an OR gate;
- a ``parallel`` group fails when all blocks fail: an AND gate;
- a ``k_of_n`` group works when at least k of its n blocks work
  (k-out-of-n:G), so it fails when at least n - k + 1 fail: a K_OF_N gate
  with that k;
- a ``switch`` is a house event: ``"open"`` cuts the path (the event has
  occurred, ``state: true``), ``"closed"`` is a wire (``state: false``);
- a parallel group of an input and a block of kind ``conditioning`` is an
  INHIBIT gate: the group fails when the input fails and the condition holds,
  and the conditioning block's ``prob`` is P(condition | input);
- ``{"ref": id}`` repeats a block or a group, as in a fault tree.

XOR (exactly one input fails) is not coherent: repairing a component can make
it fail. No arrangement of blocks has that behaviour. PRIORITY AND depends on
the order of failures, which a static diagram does not record. Both stay
fault-tree logic inside the diagram, as a ``{"fault_tree": node}`` block; such
a block can also hold any other fault-tree node.

The RBD format is a recursive JSON object marked ``"model": "rbd"``::

    {"model": "rbd", "id": "SYS", "name": "Cooling", "structure": "series",
     "blocks": [
        {"id": "PS", "name": "Power supply", "prob": 1e-4},
        {"id": "PUMPS", "structure": "parallel", "blocks": [
            {"id": "PA", "prob": 0.01}, {"id": "PB", "prob": 0.01}]},
        {"structure": "k_of_n", "k": 2, "blocks": [...]},
        {"id": "MAINT_B", "switch": "closed"},
        {"ref": "PS"},
        {"fault_tree": {"id": "G", "gate": "XOR", "children": [...]}}]}

A block's ``prob`` is the same engine input as an event's ``prob`` in a fault
tree: a failure probability, or a reliability in success mode. Blocks may
carry the other event fields (``failure_model``, ``uncertainty``,
``name_pt``...), and ``kind`` ``"undeveloped"`` or ``"conditioning"``. Groups
take ``id`` (optional: an unnamed group gets ``<parent id>.<position>``),
``name`` and ``name_pt``. The root also keeps the top event's fields
(``note``, ``mission_time``, ``time_unit``, ``prob_file``...).
"""
from copy import deepcopy
from typing import Any, Dict, Optional, Set

#: RBD group structures and the fault-tree gate each one becomes.
STRUCTURES = {"series": "OR", "parallel": "AND", "k_of_n": "K_OF_N"}
#: Block kinds (fault-tree event types of a leaf, house events aside).
BLOCK_KINDS = ("basic", "undeveloped", "conditioning")
SWITCH_STATES = {"open": True, "closed": False}

# Keys that only the RBD form or only the fault-tree form uses; every other
# key of a node (name, prob, failure_model, GUI fields...) goes across as is.
_GROUP_KEYS = {"model", "structure", "blocks", "k"}
_GATE_KEYS = {"event_type", "gate", "k", "children"}
_BLOCK_KEYS = {"model", "kind", "switch"}
_LEAF_KEYS = {"event_type", "gate", "children", "state"}
#: Keys with a meaning of their own in an RBD: never copied from a tree node.
_RBD_KEYS = {"model", "kind", "switch", "structure", "blocks", "fault_tree", "ref"}


def is_rbd(model: Any) -> bool:
    return isinstance(model, dict) and model.get("model") == "rbd"


def _item_type(item: Any, where: str) -> str:
    if not isinstance(item, dict):
        raise ValueError(f"RBD item at {where} must be a JSON object, got: {item!r}")
    tree_fields = sorted({"children", "gate", "event_type", "state"} & set(item))
    if tree_fields:
        raise ValueError(
            f"RBD item at {where} has fault-tree fields ({', '.join(map(repr, tree_fields))}); "
            "RBD groups use 'structure' and 'blocks', blocks 'kind' and 'switch', and "
            "fault-tree logic goes in a {'fault_tree': node} block")
    kinds = [key for key in ("ref", "fault_tree", "structure", "switch") if key in item]
    if len(kinds) > 1:
        raise ValueError(f"RBD item at {where} mixes {' and '.join(repr(k) for k in kinds)}")
    if kinds:
        return kinds[0]
    if "blocks" in item:
        raise ValueError(f"RBD item at {where} has 'blocks' but no 'structure'")
    return "block"


def _explicit_ids(item: Any, found: Set[str]) -> None:
    """Every id written in the diagram, fault-tree blocks included."""
    if not isinstance(item, dict):
        return
    if "id" in item:
        found.add(str(item["id"]))
    for key in ("blocks", "children"):
        for child in item.get(key) or []:
            _explicit_ids(child, found)
    _explicit_ids(item.get("fault_tree"), found)


def _holds_rbd(node: Any) -> bool:
    """Whether a fault-tree node holds RBD items (groups, blocks, switches)."""
    if not isinstance(node, dict):
        return False
    if is_rbd(node) or {"structure", "blocks", "switch"} & set(node):
        return True
    return any(_holds_rbd(child) for child in node.get("children") or [])


def rbd_to_tree(rbd: Dict[str, Any]) -> Dict[str, Any]:
    """The fault tree of an RBD (failure logic of the same system).

    The result is a recursive faultree tree, ready for :func:`normalize_tree`
    and every analysis. Ids are kept, so results, cut sets and importance
    refer to the diagram's blocks and groups."""
    if not is_rbd(rbd):
        raise ValueError("An RBD must be a JSON object with \"model\": \"rbd\"")
    explicit: Set[str] = set()
    _explicit_ids(rbd, explicit)
    generated: Dict[str, Any] = {}

    def group_id(item: Dict[str, Any], parent: Optional[str], index: int) -> str:
        if "id" in item:
            return str(item["id"])
        if parent is None:
            raise ValueError("The RBD's top group needs an 'id' (the system's id)")
        gid = f"{parent}.{index + 1}"
        # The same position again is a repeated definition of its group.
        if gid in explicit or generated.get(gid, (parent, index)) != (parent, index):
            raise ValueError(
                f"The id '{gid}' generated for an unnamed group is also used elsewhere; "
                "give that group an 'id'")
        generated[gid] = (parent, index)
        return gid

    def convert(item: Any, parent: Optional[str], index: int, where: str) -> Dict[str, Any]:
        kind = _item_type(item, where)
        if kind == "ref":
            if parent is None:
                raise ValueError("The RBD's top item cannot be a ref")
            if set(item) != {"ref"}:
                raise ValueError(
                    f"Repeated block at {where} takes only 'ref', not "
                    f"{', '.join(sorted(set(item) - {'ref'}))}; they belong to its definition")
            return {"ref": item["ref"]}
        if kind == "fault_tree":
            extra = set(item) - {"fault_tree", "model"}
            if extra:
                raise ValueError(
                    f"Fault-tree block at {where} takes only 'fault_tree'; put "
                    f"{', '.join(sorted(extra))} in the fault-tree node")
            node = item["fault_tree"]
            if not isinstance(node, dict) or "ft_nodes" in node or _holds_rbd(node):
                raise ValueError(
                    f"'fault_tree' at {where} must be a recursive fault-tree node, "
                    "without RBD groups, blocks or switches inside")
            return deepcopy(node)
        if kind == "structure":
            return convert_group(item, parent, index, where)
        if "id" not in item:
            raise ValueError(f"RBD block at {where} has no 'id'")
        node = {"id": item["id"]}
        if kind == "switch":
            state = item["switch"]
            if state not in SWITCH_STATES:
                raise ValueError(
                    f"Switch '{item['id']}' must be \"open\" (path cut: the event "
                    "occurred) or \"closed\" (a wire)")
            if "kind" in item or item.get("prob") is not None:
                raise ValueError(
                    f"Switch '{item['id']}' takes only its position, not 'kind' or 'prob'")
            node["event_type"] = "house"
            node["state"] = SWITCH_STATES[state]
        else:
            block_kind = item.get("kind", "basic")
            if block_kind not in BLOCK_KINDS:
                raise ValueError(
                    f"Block '{item['id']}' has kind {block_kind!r}; expected one of "
                    + ", ".join(BLOCK_KINDS))
            node["event_type"] = block_kind
        node["gate"] = None
        node.update({k: deepcopy(v) for k, v in item.items()
                     if k not in _BLOCK_KEYS and k != "id"})
        node["children"] = []
        return node

    def convert_group(item: Dict[str, Any], parent: Optional[str], index: int,
                      where: str) -> Dict[str, Any]:
        structure = item["structure"]
        if structure not in STRUCTURES:
            raise ValueError(
                f"RBD group at {where} has structure {structure!r}; expected one of "
                + ", ".join(STRUCTURES))
        blocks = item.get("blocks")
        if not isinstance(blocks, list) or not blocks:
            raise ValueError(f"RBD {structure} group at {where} needs a non-empty 'blocks' list")
        gid = group_id(item, parent, index)
        children = [convert(child, gid, i, f"{gid}.blocks[{i}]")
                    for i, child in enumerate(blocks)]
        conditions = [i for i, child in enumerate(blocks)
                      if isinstance(child, dict) and child.get("kind") == "conditioning"]
        gate = STRUCTURES[structure]
        if conditions:
            if structure != "parallel" or len(blocks) != 2 or len(conditions) != 1:
                raise ValueError(
                    f"A conditioning block belongs in a parallel group of two: its "
                    f"input and the condition (group '{gid}')")
            gate = "INHIBIT"
        node = {"id": item["id"]} if "id" in item else {"id": gid}
        node.update({k: deepcopy(v) for k, v in item.items()
                     if k not in _GROUP_KEYS and k != "id"})
        node["event_type"] = "top" if parent is None else "intermediate"
        node["gate"] = gate
        if structure == "k_of_n":
            k = item.get("k")
            n = len(blocks)
            if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= n:
                raise ValueError(
                    f"k_of_n group '{gid}' needs an integer 'k' (blocks that must "
                    f"work) with 1 <= k <= {n}")
            node["k"] = n - k + 1
        elif "k" in item:
            raise ValueError(f"'k' applies to k_of_n groups, not to {structure} group '{gid}'")
        node["children"] = children
        return node

    return convert(rbd, None, 0, "root")


def tree_to_rbd(tree: Dict[str, Any]) -> Dict[str, Any]:
    """The RBD of a fault tree (recursive or flat), the inverse of
    :func:`rbd_to_tree`.

    XOR and PAND gates, and a K_OF_N gate with k = 0 (always failed), have no
    block structure; they become ``{"fault_tree": node}`` blocks holding the
    gate and everything under it."""
    from .builder import _gate_semantics, _is_leaf, normalize_tree
    normalized = normalize_tree(tree)
    if normalized.get("id") is None:
        raise ValueError("The top event needs an 'id' (the RBD's top group takes it)")
    # A node written in full once; its other occurrences become refs (an
    # identical repeated definition kept as is inside a fault-tree block
    # would conflict with the one converted outside it).
    written: Set[str] = set()

    def first(node: Dict[str, Any]) -> bool:
        nid = node.get("id")
        if nid is None:
            return True
        if str(nid) in written:
            return False
        written.add(str(nid))
        return True

    def verbatim(node: Dict[str, Any]) -> Dict[str, Any]:
        if "ref" in node or not first(node):
            return {"ref": node.get("ref", node.get("id"))}
        out = {k: deepcopy(v) for k, v in node.items() if k != "children"}
        if "children" in node:
            out["children"] = [verbatim(child) for child in node["children"]]
        return out

    def convert(node: Dict[str, Any]) -> Dict[str, Any]:
        if "ref" in node:
            return {"ref": node["ref"]}
        if _is_leaf(node):
            if node.get("id") is None:
                raise ValueError(f"Event without an 'id': {node!r}")
            if not first(node):
                return {"ref": node["id"]}
            item = {"id": node["id"]}
            event_type = node.get("event_type")
            if event_type == "house":
                item["switch"] = "open" if node["state"] else "closed"
            elif event_type in ("undeveloped", "conditioning"):
                item["kind"] = event_type
            item.update({k: deepcopy(v) for k, v in node.items()
                         if k not in _LEAF_KEYS and k not in _RBD_KEYS and k != "id"
                         and not (event_type == "house" and k == "prob" and v is None)})
            return item
        gate = node.get("gate")
        children = node.get("children", [])
        k = node.get("k")
        if gate not in ("XOR", "PAND"):
            _gate_semantics(node, len(children), False)  # rejects a bad gate or k
        if not first(node):
            return {"ref": node["id"]}
        if gate in ("XOR", "PAND") or (gate == "K_OF_N" and k == 0):
            if node.get("id") is not None:
                written.discard(str(node["id"]))
            return {"fault_tree": verbatim(node)}
        structure = {"OR": "series", "AND": "parallel", "INHIBIT": "parallel",
                     "K_OF_N": "k_of_n"}.get(gate)
        if structure is None:
            raise ValueError(f"Unsupported gate: {gate}")
        item = {} if node.get("id") is None else {"id": node["id"]}  # rbd_to_tree names it
        item.update({k_: deepcopy(v) for k_, v in node.items()
                     if k_ not in _GATE_KEYS and k_ not in _RBD_KEYS and k_ != "id"})
        item["structure"] = structure
        if gate == "K_OF_N":
            item["k"] = len(children) - k + 1
        item["blocks"] = [convert(child) for child in children]
        return item

    return {"model": "rbd", **convert(normalized)}
