"""Sanity-check the ported workplace_assistant verifier against the dataset.
For each row: gt-vs-gt must be True; gt-vs-[] must be False (task actually changes state).
Also counts ground-truth calls that raise when executed."""
import json, sys, time, contextlib, io
from rl.workplace_assistant.utils import is_correct

def main(path, limit):
    rows = [json.loads(l) for l in open(path)][:limit]
    t0 = time.time(); self_fail = noop = errs = 0
    for r in rows:
        gt = r["ground_truth"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ok = is_correct(gt, gt, None)
            empty = is_correct([], gt, None)
        errs += "Error executing tool" in buf.getvalue()
        self_fail += (not ok); noop += bool(empty)
    n = len(rows)
    print(f"{path.split('/')[-1]}: rows={n} self_fail={self_fail} noop_tasks={noop} rows_with_gt_errors={errs} secs={time.time()-t0:.1f}")

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10**9)
