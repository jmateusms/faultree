# Faultree

A Python tool for Fault Tree Analysis (FTA) using Ordered Binary Decision Diagrams (OBDD).

## Overview
- Converts a Fault Tree (FTA) JSON into an Ordered BDD using the `dd` package.
- Supports logic gates: AND, OR, XOR (exactly-one), K-of-N.
- Exact top-event and intermediate-event probabilities in O(|BDD|) time
  (memoized Shannon cofactor recursion) — repeated/shared events are handled
  exactly, with no cut-set approximations.
- Outputs algebraic and symbolic expressions.
- Supports internal event cloning via `ref` (reference nodes), making the
  model a DAG.
- Probabilities can be scalars or sample arrays (evaluated elementwise), given
  inline, as a JSON mapping, or loaded from CSV/Excel files.

## Installation

Requires Python ≥ 3.10.

```bash
pip install faultree            # library + CLI (needs only dd and numpy)
pip install "faultree[server]"  # + FastAPI/uvicorn for the API server
pip install "faultree[excel]"   # + pandas/openpyxl to read Excel probability files
```

CSV probability files need no extra dependency. faultree uses `dd.autoref`,
the pure-Python BDD backend of `dd`, so no C compiler or CUDD build is
needed.

From a checkout of this repository:

```bash
pip install -e ".[server,test]"
```

## Tree Format
- **Fields**:
  - `id` (string): Unique identifier for the event.
  - `name` (string): Descriptive name.
  - `event_type`: `top`, `intermediate`, `basic`, or `undeveloped`.
  - `gate`: `AND`, `OR`, `XOR`, `K_OF_N`, or `null`/`BASIC` for leaves.
  - `children` (list): Child nodes.
  - `k` (int): Required for `K_OF_N` gates (`0 <= k <= n`).
  - `prob` (finite float or non-empty list of finite floats): Probability for
    basic/undeveloped events. A list is treated as a one-dimensional sample
    vector and propagated elementwise.
  - `ref` (string): ID of another node to clone/reference.
  - `prob_file` (string, optional, top level): CSV/Excel file with one column
    per basic-event id, resolved relative to the tree JSON.

Gate semantics: `XOR` means **exactly one** input (a mutually-exclusive gate),
not chained parity. `K_OF_N` means **at least k** of the n inputs.

Every basic event must have a probability (from the tree, `--probs`, or a
probability file); a missing probability is an error unless
`--assume-missing-zero` is passed.

## Supported Formats
Faultree supports two JSON formats (auto-detected):
1. **Recursive Tree** (standard): Nodes nested within `children`.
2. **Flat List**: Nodes defined in `ft_nodes` and `be_nodes` lists, with
   `branches` referencing child IDs.

Event IDs must identify one definition. Repeating an identical definition is
accepted as an inline copy of the same event; conflicting repeated definitions
are rejected. A flat model must provide an explicit `analysis`/`esd_nodes`
root or have exactly one unreferenced root, and every declared node must be
reachable from it.

## Usage

```bash
faultree examples/basic_tree.json
# or equivalently
python -m faultree examples/basic_tree.json
```

Override probabilities (inline JSON or a CSV/Excel path):

```bash
faultree examples/basic_tree.json --probs '{"BE1": 0.05}'
faultree examples/fta4b.json --probs examples/fta4b_probs.csv
```

Reliability analysis (dual/success tree). The tree keeps its failure-logic
structure, but inputs are interpreted as reliabilities and the result is the
system success probability `R = 1 - Q`:

```bash
faultree examples/fta3_success.json --reliability
```

Structured JSON (top-event probability, conditional probabilities, importance
measures and the assumptions used) and minimal cut sets:

```bash
faultree examples/fta4.json --structured
faultree examples/fta4.json --cut-sets        # optional max order, default 6
```

JSON output is strict: non-finite values (for example an infinite risk
reduction worth when one event alone guards the top event) are written as
`null`.

Run the API server (binds to `127.0.0.1` by default; pass `--host 0.0.0.0`
to expose it):

```bash
faultree --serve
```

**Example request**:

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "tree": {
      "id": "TOP",
      "gate": "OR",
      "children": [
        {"id": "A", "prob": 0.1},
        {"id": "B", "prob": 0.2}
      ]
    }
  }'
```

## Features
- **Logic Gates**: AND, OR, XOR (exactly one), K-of-N (at least k).
- **Exact quantification**: weighted model counting via the BDD cofactor
  recursion — linear in BDD size, exact for repeated/shared events.
- **Importance measures** (`analyze()`, `--structured`; fault-tree mode):
  Birnbaum, criticality `(Q − Q0)/Q`, risk achievement worth (RAW) and risk
  reduction worth (RRW), computed exactly on the BDD for every basic event,
  also elementwise for sample vectors. Criticality equals the risk-decrease
  form that PRA codes often call Fussell-Vesely (`1 − 1/RRW`); Fussell's
  cut-set definition is larger and agrees with it only for rare events.
- **Minimal cut sets** (`minimal_cut_sets()`, `--cut-sets`): bounded
  enumeration for monotone AND/OR/K-of-N trees, with an explicit
  completeness flag.
- **Dual tree mode**: reliability/success probability from the same tree.
- **Sampled probabilities**: equal-length array-valued probabilities are joint
  samples and propagate elementwise. Unequal lengths are rejected by default;
  `--resample-independent` explicitly bootstraps independent marginals and
  therefore discards joint alignment. `--shuffle` uses a shared permutation.
- **Symbolic output**: algebraic expressions (e.g. `(A + B * C)`).
- **Ref/clones**: reuse events within the same tree using `{"ref": "ID"}`
  (cycles are detected and rejected).
- **API**: FastAPI server (`/analyze`, `/health`) for integrating with other
  tools.

## Development

```bash
pip install -e ".[server,test]"
python -m pytest
```

`tests/test_auto_examples.py` cross-checks every example tree against an
independent brute-force oracle; `tests/test_examples.py` pins hand-computed
golden values; `tests/test_fixes.py` covers regression cases (success-mode
XOR duality, cycle detection, input validation, server behavior).

## Backlog
- **Minimal Cut Sets at scale**: ZBDD-based extraction beyond the bounded
  enumeration.
- **Variable Ordering**: Heuristics and `dd` sifting for BDD size reduction.
- **Time-Dependent Analysis**: Exponential/Weibull distributions,
  availability.
- **Sound Uncertainty Propagation**: jointly sampled Monte Carlo / LHS with
  percentile bounds (replacing per-event resampling).
- **Common-Cause Failure**: beta-factor / MGL / alpha-factor groups.
- **Transfer Symbols**: Split trees across multiple files (`transfer_in`).
- **Open-PSA MEF**: import/export for ecosystem interoperability.
- **Frontend**: Web UI for visualization and analysis.

## Notes
- Assumes independent basic events.
- Variable ordering is based on traversal order unless specified with
  `--ordering` (which must cover all basic events).
- The engine (and the underlying `dd` library) is recursive: extremely deep
  trees/BDDs (~1000 levels) can hit Python's recursion limit.

### Bounded monotone minimal cuts

`minimal_cut_sets(tree, max_order=6, max_candidates=10000, max_sets=1000,
max_basic_events=32)` returns positive minimal cuts for AND/OR/K_OF_N models.
Always inspect `complete`, `truncated` and `reason`; an incomplete list is not
an exhaustive reliability result. XOR needs signed implicants and is rejected.
Limits apply to enumeration, not a hard wall-clock or BDD-memory budget.
