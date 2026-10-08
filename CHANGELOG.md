# Changelog

## Unreleased

### Added
- `faultree gui` (also `python -m faultree gui`, options `--port`,
  `--no-browser`, `--host`): a local graphical interface served by a
  standard-library HTTP server on 127.0.0.1, with no new dependency and
  nothing loaded from the network. Tree editor (AND, OR, XOR, k-of-n, shared
  events, `ref` clones, recursive and flat JSON), probability files, input
  distributions (lognormal by median and error factor, beta, uniform,
  log-uniform) sampled with a seed and propagated as joint sample vectors,
  and a results dashboard: exact Q/R, probabilities on the tree, minimal cut
  sets with completeness flag, importance bars and scatter, what-if
  (conditional Q), distribution of Q with percentiles, importance spread, the
  expressions and a BDD drawing. Results export as JSON and charts as SVG or
  PNG. Interface in Portuguese and English.
- GUI failure models: an event's probability can come from a time-to-failure
  distribution at a mission time t (exponential, Weibull, normal, lognormal,
  gamma, uniform: p = F(t)) or from a count of failures (binomial over n
  demands, Poisson over [0, t], reaching k), with the reliability 1 − F(t) in
  success mode. Saved as `failure_model` with `mission_time`/`time_unit`, and
  `prob` = F(t) so the engine and CLI read the file unchanged.
  `examples/cooling_system.json` uses every kind.
- GUI shared-event markers: every occurrence of a shared event, and a gate
  with its clones, carries the same letter and colour (or the colour alone,
  or ×n); hovering or selecting one highlights the others.
- GUI settings (⚙): marker style, occurrence highlighting and scientific
  notation, remembered in the browser.
- `examples/pressure_tank.json`: the pressure tank of the NRC Fault Tree
  Handbook (NUREG-0492), with illustrative probabilities and uncertainty.
- The examples are packaged as `faultree/examples` (for the GUI's examples
  menu in non-editable installs).
- `analyze()` returns a structured, reproducible result: top-event `Q`,
  per-event probabilities, `conditional_Q` (top event with each basic event
  forced true/false), signed Birnbaum importance and the assumptions used
  (variable order, seed, sample handling, version). Also `--structured` in
  the CLI and a `result` field in the API response.
- `importance` in `analyze()`: criticality `(Q − Q0)/Q` (the risk-decrease
  form often reported as Fussell-Vesely, not the cut-set FV), RAW and RRW
  next to Birnbaum, exact on the BDD and elementwise for sample vectors. Not
  defined (`None`) in success mode.
- `compile_tree()` / `quantify_compiled()` to compile once and quantify many
  probability sets; `benchmark_orderings()` to measure caller-given orders.
- `minimal_cut_sets()` for monotone AND/OR/K-of-N trees, bounded and with an
  explicit `complete`/`truncated` flag; `--cut-sets [MAX_ORDER]` in the CLI.

### Fixed
- The GUI's examples menu keeps the loaded example when the language changes.
- Probability CSV files separated by `;` (with decimal commas, as Excel
  writes them in pt-BR) or by tabs are read instead of failing with a
  misleading "more cells than header columns" error.
- The `.csv`/`.xls`/`.xlsx` extension check is no longer case-sensitive.

### Changed
- pandas and openpyxl are no longer required: CSV probability files are read
  with the standard library, and Excel files need the new `faultree[excel]`
  extra (a clear ImportError says so). A plain install is now dd + numpy.
- Stricter model contract: conflicting repeated definitions of an event id
  are rejected; flat models need an explicit or unique reachable root;
  probabilities must be finite scalars or non-empty 1-D vectors; unequal
  sample-vector lengths are rejected unless `resample_independent=True`.
- Structured JSON from the CLI and API is strict: non-finite numbers are
  written as `null`.

## 0.3.0 — 2026-08-04

Engine correctness and performance release (Phase 0 of the upgrade roadmap).
All example golden values are unchanged; the fixes below change *when errors
are raised* and *one wrong number* (success-mode XOR), not any previously
correct result.

### Fixed
- **Quantification is now O(|BDD|)** — `wmc()` uses the memoized Shannon
  cofactor recursion `pr(f) = q·pr(f_high) + (1−q)·pr(f_low)` (with correct
  complemented-edge handling) instead of enumerating satisfying assignments,
  which was exponential. A 20-variable parity BDD (211 nodes) went from
  ~3.9 s to ~0.1 ms. Complement edges are pushed into the children rather
  than computed as `1 − r`, preserving full *relative* accuracy for
  rare-event probabilities (verified ≤ 2e-16 relative error down to 1e-18;
  an adversarial review caught that the naive complement handling floored
  such values to 0.0).
- **Success/reliability mode XOR** returned a wrong number for gates with
  ≥ 3 inputs (e.g. 0.908 instead of the correct 0.602). The dual of
  "exactly one of n fails" is now correctly "NOT exactly n−1 succeed".
  The gate/dual dispatch now lives in a single function
  (`_gate_semantics`) shared by the BDD, expression, and symbolic
  traversals so the three renderings cannot drift apart again.
- **Server returned `top_event_probability: null` for flat-format trees**
  (`ft_nodes`/`be_nodes`); the tree is now normalized before the top-event
  id lookup.
- **Server 500 on array probabilities** — numpy arrays/scalars in results
  are converted to native JSON types.

### Changed (stricter input handling)
- A basic event with **no probability now raises** a `ValueError` listing the
  missing ids instead of silently contributing probability 0.0. Opt out with
  `allow_missing=True` / `--assume-missing-zero`.
- Probabilities are validated to be within **[0, 1]**.
- **Cyclic `ref` / flat-format `branches` structures raise** a clear
  "Cycle detected: A -> B -> A" error instead of overflowing the stack.
- An explicit `--ordering` that misses basic events (or contains duplicates)
  raises a clear error instead of a raw `dd` internal error.
- Gates with **no children** and `K_OF_N` with **k > n** raise instead of
  silently evaluating to constants.
- Sample-array resampling/shuffling is now **seeded (default seed 0,
  reproducible)**; pass `--seed N` / `seed=None` to change. Note: ragged
  per-event sample arrays are still resampled independently — use
  equal-length jointly drawn samples when correlations matter.
- Server errors are split into 400 (invalid input) vs 500 (internal, logged);
  added a `GET /health` endpoint; server/CLI default host changed from
  `0.0.0.0` to `127.0.0.1`.

### Packaging
- `setup.py` removed; `pyproject.toml` is the single source of packaging
  metadata. Requires Python ≥ 3.10 (the `dd` dependency has no installable
  wheel/sdist path on 3.9 with current setuptools).
- GitHub Actions CI (Python 3.10–3.13) added.
- `tests/auto_examples.py` renamed to `tests/test_auto_examples.py` so the
  brute-force-oracle suite is collected by default pytest discovery; its
  success-mode oracle now uses the definitional dual (complement of the
  failure tree) and covers XOR/K-of-N examples in success mode.
- `examples/bdd-resultado-52.json` (a generated result dump, not an input
  fixture) removed from the repository.

## 0.2.1 — 2026-04-01

- Package overhaul: OBDD-based solving via the `dd` package (`builder.py`),
  CLI (`faultree`), FastAPI server (`--serve`), recursive and flat JSON
  input formats, `ref` clone nodes, success/dual-tree mode, scalar and
  sampled (array) probabilities, CSV/Excel probability files, example suite.
  Published to PyPI. (0.2.0 was never published.)

## 0.1.0 — 2025-03-21

- Initial release.
