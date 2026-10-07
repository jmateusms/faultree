import argparse
import json
import sys
import os
import numpy as np
from .builder import analyze, build, compute_event_probabilities, normalize_tree, to_jsonable
from .cutsets import minimal_cut_sets


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["gui"]:
        # "faultree gui [--port N] [--no-browser]": the graphical interface.
        # (A tree file named "gui" can still be passed as ./gui.)
        from .gui.server import main as gui_main
        sys.exit(gui_main(argv[1:]))

    p = argparse.ArgumentParser(
        description="Build OBDD from FTA JSON and evaluate probabilities.",
        epilog="Graphical interface: faultree gui [--port N] [--no-browser]")
    p.add_argument("json_file", nargs='?', help="Path to FTA JSON file")
    p.add_argument("--ordering", nargs='*', default=None, help="Optional variable ordering (basic event ids)")
    p.add_argument("--probs", type=str, default=None, help="JSON mapping of basic event id -> probability, or path to CSV/Excel file")
    p.add_argument("--use-names", action="store_true", help="Use event names instead of IDs in expression output")
    p.add_argument("--reliability", action="store_true", help="Calculate Reliability (Success Probability) from a Fault Tree (Dual Tree Mode)")
    p.add_argument("--shuffle", action="store_true", help="Apply one shared shuffle to aligned sample vectors")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sample shuffling or explicit resampling (default: 0)")
    p.add_argument("--resample-independent", action="store_true", help="Explicitly bootstrap unequal sample vectors as independent marginals (destroys joint alignment)")
    p.add_argument("--assume-missing-zero", action="store_true", help="Treat basic events with no probability as 0.0 instead of raising an error")
    p.add_argument("--structured", action="store_true", help="Print structured JSON with Q, conditional Q, importance measures (Birnbaum, criticality, RAW, RRW), and assumptions")
    p.add_argument("--cut-sets", nargs="?", const="6", default=None, metavar="MAX_ORDER",
                   help="Print the minimal cut sets as JSON (monotone AND/OR/K_OF_N trees; default max order 6) and exit")
    p.add_argument("--serve", action="store_true", help="Run the API server")
    p.add_argument("--host", default="127.0.0.1", help="API host (default: 127.0.0.1; use 0.0.0.0 to expose on all interfaces)")
    p.add_argument("--port", type=int, default=8000, help="API port (default: 8000)")

    args = p.parse_args(argv)

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

    if args.cut_sets is not None:
        try:
            args.cut_sets = int(args.cut_sets)
        except ValueError:
            # "faultree --cut-sets tree.json": the optional MAX_ORDER swallowed
            # the file name, so give it back and use the default order.
            if args.json_file:
                p.error(f"argument --cut-sets: invalid int value: {args.cut_sets!r}")
            args.json_file, args.cut_sets = args.cut_sets, 6
        if args.reliability:
            p.error("--cut-sets lists failure cut sets of the fault tree;"
                    " it cannot be combined with --reliability")

    if not args.json_file:
        p.error("the following arguments are required: json_file (unless --serve is used)")

    try:
        with open(args.json_file, "r") as fh:
            tree = json.load(fh)
    except OSError as e:
        sys.exit(f"Error: cannot read '{args.json_file}': {e}")
    except json.JSONDecodeError as e:
        sys.exit(f"Error: '{args.json_file}' is not valid JSON: {e}")

    if not args.structured and args.cut_sets is None:
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

        if args.cut_sets is not None:
            result = minimal_cut_sets(tree, max_order=args.cut_sets)
            print(json.dumps(result, indent=2))
            return

        if args.structured:
            result = analyze(
                tree, prob_input, args.ordering, use_names=args.use_names,
                success_mode=args.reliability, shuffle=args.shuffle,
                seed=args.seed, allow_missing=args.assume_missing_zero,
                resample_independent=args.resample_independent)
            print(json.dumps(to_jsonable(result), indent=2, sort_keys=True))
            return

        bdd, top, expr, symb = build(tree, args.ordering, use_names=args.use_names, success_mode=args.reliability)
        print("Expression:")
        print(expr)
        print("Symbolic:")
        print(symb)

        prob_map = compute_event_probabilities(
            bdd, tree, prob_input,
            success_mode=args.reliability, shuffle=args.shuffle,
            seed=args.seed, allow_missing=args.assume_missing_zero,
            resample_independent=args.resample_independent)
    except (ValueError, KeyError, TypeError, ImportError) as e:
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
