# Faultree

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23244049.svg)](https://doi.org/10.5281/zenodo.23244049)

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

## Graphical interface

`faultree gui` (or `python -m faultree gui`) opens a local web app in the
browser: load or build a tree, give the basic events probabilities or
distributions, and explore the exact results without writing code.

```bash
faultree gui                            # http://127.0.0.1:8765/, or a free port
faultree gui --port 9000 --no-browser   # choose the port; open the page yourself
```

It runs entirely on your computer: a small standard-library server, bound to
127.0.0.1, serves the page and calls `analyze()` and `minimal_cut_sets()`.
Nothing is loaded from the internet (it works offline) and it needs no extra
dependency. Ctrl+C stops it. The interface is in Portuguese and English: it
follows the browser's language, and the PT/EN switch changes it.

![The pressure tank in the GUI: model on the left, probabilities on the tree on the right](https://raw.githubusercontent.com/jmateusms/faultree/main/docs/gui/tree.png)

The model is on the left:

- **Tree**: the editor. Select a node to add basic events or gates, change
  the gate (AND, OR, XOR = exactly one, k-out-of-n), share a basic event
  between branches, or add a clone of a gate (faultree's `ref`); undo and
  redo with Ctrl+Z / Ctrl+Shift+Z. Every occurrence of a shared event, and a
  gate together with its clones, carries the same marker in a colour of its
  own (a letter A, B, C…, the colour alone, or the count ×n), and hovering or
  selecting one highlights all the others.
- **Probabilities**: the probability of every basic event, given directly or
  by a failure model at a mission time t: p = F(t) for a time to failure that
  is exponential, Weibull, normal, lognormal, gamma or uniform, or a count of
  failures reaching k (binomial over n demands, Poisson over [0, t]). In
  success mode the event value is the reliability 1 − F(t). Optionally, a
  distribution of p for the uncertainty analysis (lognormal by median and
  error factor, beta, uniform, log-uniform); a CSV/Excel probability file can
  be imported (one column per event; one row gives point values, several rows
  give joint sample vectors).
- **JSON**: the model in faultree's format, to edit or paste (recursive or
  flat) and apply.
- The top bar opens the examples in `examples/` and JSON files, saves the
  model in the recursive or the flat format, and switches between failure
  analysis (Q) and success analysis (R, with reliabilities as inputs). The
  ⚙ button opens the settings: the shared-event marker, highlighting of
  occurrences and automatic or scientific notation (kept in the browser).

The results are on the right, recomputed as you edit:

- A summary: exact Q (or R) and its complement, the number of minimal cut
  sets with the completeness flag, the model and BDD sizes, and the
  uncertainty band once it has been run.
- **Tree**: the exact probability of every event drawn on the tree (one-hue,
  logarithmic colour scale), with the values on hover.
- **Minimal cut sets**: a sortable table (order, probability, share of Q),
  the largest contributions, the completeness flag and maximum order, and the
  rare-event sum and min-cut upper bound next to the exact Q. Hovering a row
  marks its events on the model tree.
- **Importance**: Birnbaum, criticality, RAW and RRW of every basic event as
  sortable bars, and a RAW × RRW (or Birnbaum × criticality) scatter with the
  usual screening thresholds.
- **What if?**: click basic events to set them failed or working; Q and every
  gate are recomputed exactly, with the ratio to the base Q.
- **Uncertainty**: draws joint samples of the events that have a
  distribution, propagates them exactly as sample vectors, and shows the
  histogram or CDF of Q with its 5/50/95 % percentiles, mean and point value,
  and the spread of each importance measure per event. Runs are reproducible:
  each event draws from `numpy.random.default_rng([seed, zlib.crc32(event_id)])`.
- **Expression and BDD**: faultree's expression and symbolic forms and, for
  small trees, the BDD with the probability of each node (the Shannon
  recursion behind the exact result).
- **Export results** saves a JSON with the model, settings, point results,
  what-if scenario and uncertainty run; every chart has SVG and PNG buttons.

The GUI keeps an event's distribution in an `uncertainty` field
(`{"dist": "lognormal", "median": 1e-3, "ef": 3}`), its failure model in
`failure_model` (`{"dist": "weibull", "beta": 1.8, "eta": 12000}`) with
`mission_time` and `time_unit` on the top event, and an optional Portuguese
name in `name_pt`; the engine ignores them and reads `prob`, which the GUI
saves as F(t) at the mission time. `examples/pressure_tank.json` is the
pressure tank of the NRC Fault Tree Handbook (NUREG-0492), with illustrative
probabilities; `examples/cooling_system.json` uses every kind of failure
model. More screenshots are in [docs/gui](https://github.com/jmateusms/faultree/tree/main/docs/gui).

## Installation

Requires Python ≥ 3.10.

```bash
pip install faultree            # library + CLI (needs only dd and numpy)
pip install "faultree[server]"  # + FastAPI/uvicorn for the API server
pip install "faultree[excel]"   # + pandas/openpyxl to read Excel probability files
```

CSV probability files need no extra dependency. faultree uses `dd.autoref`,
the pure-Python BDD backend of `dd`, so no C compiler or CUDD is needed at
run time, which also lets it run in the browser under Pyodide. PyPI has only
Linux wheels and the source of `dd`; for Pyodide, build a pure-Python wheel
first with `pip wheel dd --no-deps --no-binary dd`.

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

A probability CSV has a header row of basic-event ids and one row per sample.
Columns may be separated by `,`, `;` or a tab; with `;` or a tab, decimal
commas (`0,05`) are read as well, which is what Excel writes in pt-BR and
other comma-decimal locales. The `.csv`/`.xlsx` extension is not
case-sensitive.

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
- **GUI**: `faultree gui`, a local web interface (standard library only).

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
- **Time-Dependent Analysis**: the GUI computes event probabilities from
  failure models at one mission time; next: Q(t) curves, repairable
  components (availability), failure models in the engine and CLI, and
  uncertainty on model parameters (e.g. on λ).
- **Sound Uncertainty Propagation**: jointly sampled Monte Carlo / LHS with
  percentile bounds (replacing per-event resampling).
- **Common-Cause Failure**: beta-factor / MGL / alpha-factor groups.
- **Transfer Symbols**: Split trees across multiple files (`transfer_in`).
- **Open-PSA MEF**: import/export for ecosystem interoperability.
- **Frontend**: the local GUI (`faultree gui`) covers editing, exact results,
  cut sets, importance, what-if and uncertainty. Next: navigation of large
  trees (collapse branches, search), state-of-knowledge correlation between
  identical components in the uncertainty analysis, and a Pyodide build that
  needs no local server.

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

## Citing

If you use faultree in academic work, please cite it: GitHub's **Cite this repository** button (from [CITATION.cff](https://github.com/jmateusms/faultree/blob/main/CITATION.cff)) gives APA and BibTeX. Releases are archived on Zenodo: [doi:10.5281/zenodo.23244049](https://doi.org/10.5281/zenodo.23244049) cites the software across versions, and each release has its own DOI on that page.

## License

BSD 3-Clause — see [LICENSE](https://github.com/jmateusms/faultree/blob/main/LICENSE).
