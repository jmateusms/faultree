import argparse
import json
import sys
import os
import numpy as np
from .builder import build, compute_event_probabilities, normalize_tree

def main() -> None:
    p = argparse.ArgumentParser(description="Build OBDD from FTA JSON and evaluate probabilities.")
    p.add_argument("json_file", nargs='?', help="Path to FTA JSON file")
    p.add_argument("--ordering", nargs='*', default=None, help="Optional variable ordering (basic event ids)")
    p.add_argument("--probs", type=str, default=None, help="JSON mapping of basic event id -> probability, or path to CSV/Excel file")
    p.add_argument("--use-names", action="store_true", help="Use event names instead of IDs in expression output")
    p.add_argument("--reliability", action="store_true", help="Calculate Reliability (Success Probability) from a Fault Tree (Dual Tree Mode)")
    p.add_argument("--shuffle", action="store_true", help="Shuffle samples before processing (for array inputs)")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sample resampling/shuffling (default: 0, reproducible)")
    p.add_argument("--assume-missing-zero", action="store_true", help="Treat basic events with no probability as 0.0 instead of raising an error")
    p.add_argument("--serve", action="store_true", help="Run the API server")
    p.add_argument("--host", default="127.0.0.1", help="API host (default: 127.0.0.1; use 0.0.0.0 to expose on all interfaces)")
    p.add_argument("--port", type=int, default=8000, help="API port (default: 8000)")

    args = p.parse_args()

    if args.serve:
        try:
            from .server import run_server
        except ImportError:
            print("Error: FastAPI or Uvicorn not installed.")
            print("To use the server feature, install with: pip install faultree[server]")
            sys.exit(1)
        print(f"Starting Faultree API server on {args.host}:{args.port}...")
        run_server(host=args.host, port=args.port)
        return

    if not args.json_file:
        p.error("the following arguments are required: json_file (unless --serve is used)")

    try:
        with open(args.json_file, "r") as fh:
            tree = json.load(fh)
    except OSError as e:
        sys.exit(f"Error: cannot read '{args.json_file}': {e}")
    except json.JSONDecodeError as e:
        sys.exit(f"Error: '{args.json_file}' is not valid JSON: {e}")

    if args.reliability:
        print("Mode: Reliability Analysis (Dual Tree)")
    else:
        print("Mode: Fault Tree Analysis")

    prob_input = None
    if args.probs:
        if args.probs.strip().startswith('{'):
             try:
                 prob_input = json.loads(args.probs)
             except json.JSONDecodeError as e:
                 sys.exit(f"Error: --probs looks like JSON but failed to parse: {e}")
        else:
             prob_input = args.probs

    try:
        tree = normalize_tree(tree)

        if prob_input is None and "prob_file" in tree:
            prob_file = tree["prob_file"]
            # Try to resolve relative to json file
            json_dir = os.path.dirname(os.path.abspath(args.json_file))
            potential_path = os.path.join(json_dir, prob_file)
            if os.path.exists(potential_path):
                prob_input = potential_path
            else:
                prob_input = prob_file

        bdd, top, expr, symb = build(tree, args.ordering, use_names=args.use_names, success_mode=args.reliability)
        print("Expression:")
        print(expr)
        print("Symbolic:")
        print(symb)

        prob_map = compute_event_probabilities(
            bdd, tree, prob_input,
            success_mode=args.reliability, shuffle=args.shuffle,
            seed=args.seed, allow_missing=args.assume_missing_zero)
    except (ValueError, KeyError, TypeError) as e:
        sys.exit(f"Error: {e}")
    except RecursionError:
        sys.exit("Error: tree is too deep for the recursive engine"
                 " (Python recursion limit); simplify the tree or raise"
                 " sys.setrecursionlimit")

    if "id" not in tree:
        sys.exit("Error: tree has no top-level 'id' field")
    top_id = str(tree["id"])
    top_val = prob_map[top_id]

    if isinstance(top_val, (np.ndarray, list)):
        print(f"Top event probability [{top_id}]: {top_val}")
    else:
        print(f"Top event probability [{top_id}]: {top_val:.6g}")

    print("Intermediate/basic event probabilities:")
    for nid, val in prob_map.items():
        if nid == top_id:
            continue
        if isinstance(val, (np.ndarray, list)):
             print(f"  {nid}: {val}")
        else:
             print(f"  {nid}: {val:.6g}")


if __name__ == "__main__":
    main()
