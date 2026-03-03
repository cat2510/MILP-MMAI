#!/usr/bin/env python3
import argparse, os, sys, runpy, glob
import gurobipy as gp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help=".../processed/LogiOR")
    ap.add_argument("--out", required=True, help="where to put MPS files")
    ap.add_argument("--pattern", default="prob_*", help="folder glob, default prob_*")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # monkey-patch optimize
    _orig_optimize = gp.Model.optimize

    def _patched_optimize(self, callback=None):
        # infer problem id from CWD or an env var
        prob_id = os.environ.get("PROB_ID", "unknown")
        out_path = os.path.join(args.out, f"{prob_id}.mps")
        self.update()
        self.write(out_path)
        print(f"[OK] wrote {out_path} (skipped solve)")
        # IMPORTANT: stop execution cleanly
        raise SystemExit(0)

    gp.Model.optimize = _patched_optimize

    prob_dirs = sorted(glob.glob(os.path.join(args.root, args.pattern)))
    for d in prob_dirs:
        code_path = os.path.join(d, "code.py")
        if not os.path.isfile(code_path):
            print(f"[MISS] {d}: no code.py")
            continue

        prob_id = os.path.basename(d)  # prob_019, etc.
        os.environ["PROB_ID"] = prob_id

        try:
            # run as if `python code.py`
            runpy.run_path(code_path, run_name="__main__")
        except SystemExit:
            # expected path (we exit after writing)
            continue
        except Exception as e:
            print(f"[ERR] {prob_id}: {e}")

    # restore (optional)
    gp.Model.optimize = _orig_optimize

if __name__ == "__main__":
    main()