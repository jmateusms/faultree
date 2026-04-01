# Faultree

A Python tool for Fault Tree Analysis (FTA) using Ordered Binary Decision Diagrams (OBDD).

## Overview
- Converts a Fault Tree (FTA) JSON into an Ordered BDD using the `dd` package.
- Supports logic gates: AND, OR, XOR, K-of-N.
- Outputs algebraic and symbolic expressions.
- Computes Top Event and Intermediate Event probabilities from Basic Event probabilities.
- Supports internal event cloning via `ref` (reference nodes).

## Tree Format
- **Fields**: 
  - `id` (string): Unique identifier for the event.
  - `name` (string): Descriptive name.
  - `event_type`: `top`, `intermediate`, `basic`, or `undeveloped`.
  - `gate`: `AND`, `OR`, `XOR`, `K_OF_N`, or `null`/`BASIC` for leaves.
  - `children` (list): Child nodes.
  - `k` (int): Required for `K_OF_N` gates.
  - `prob` (float): Probability for basic/undeveloped events.
  - `ref` (string): ID of another node to clone/reference.

## Supported Formats
Faultree supports two JSON formats:
1. **Recursive Tree** (Standard): Nodes nested within `children`.
2. **Flat List** (New): Nodes defined in `ft_nodes` and `be_nodes` lists, with `branches` referencing child IDs. This format is automatically detected.

## Usage
1. **Install Dependencies**:
   ```bash
   python3 -m venv .venv
   . .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run Analysis**:
   ```bash
   python faultree_cli.py examples/sample_tree.json
   ```

3. **Override Probabilities** (optional):
   ```bash
   python faultree_cli.py examples/sample_tree.json --probs '{"BE1": 0.05}'
   ```

4. **Reliability Analysis (Success Mode)**:
   Calculate reliability (Success Probability) from a Fault Tree model. This mode interprets the input tree as a Fault Tree (Failure Logic) but computes the Dual (Success) Tree and expects reliability values as input.
   ```bash
   python faultree_cli.py examples/sample_tree_success.json --reliability
   ```

5. **Run API Server**:
   ```bash
   python faultree_cli.py --serve
   ```
   
   **Example Request**:
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
- **Logic Gates**: AND, OR, XOR (Exactly One), K-of-N (At least K).
- **Probability Calculation**: Uses Weighted Model Counting (WMC) on the BDD.
- **Symbolic Output**: Prints algebraic expressions (e.g., `(A + B * C)`).
- **Ref/Clones**: Supports reusing events within the same tree using `{"ref": "ID"}`.
- **API**: Built-in FastAPI server for integrating with other tools.

## Backlog
- **Transfer Symbols**: Support splitting trees across multiple files/trees (e.g., `transfer_in` pointing to a separate file).
- **Minimal Cut Sets**: Extract and report minimal cut sets.
- **Variable Ordering**: Implement heuristics (sifting, metaheuristics) for BDD optimization.
- **Advanced Distributions**: Support `dist`/`parameters` (Exponential, Weibull) for time-dependent analysis.
- **Data Sampling**: Estimate probabilities from sample data.
- **Frontend**: Web UI for visualization and analysis.

## Notes
- Assumes independent basic events.
- Variable ordering is currently based on traversal order unless specified.
