"""Verify [PG weights] lines in a real training log against the closed form.

ProgressiveGamma.weights() returns (gamma, gamma if mode=='both' else base_gamma)
with gamma = base_gamma * (start_ratio + (end_ratio-start_ratio)*iter/(T-1)).
ProgressiveGammaHook logs sup2=gamma and unsup2=unsup_weight*multiplier every
`log_interval` iterations, at iter 0, and at the final iteration.

Usage:
    python tools/verify_pg_weights.py <run-dir-log> --mode sup2|both \
        --total-iters 32000 --base-gamma 0.2 --unsup-weight 2.0 \
        --start-ratio 0.5 --end-ratio 1.0
"""

import argparse
import re
import sys

LINE = re.compile(
    r"\[PG weights\] iter=(\d+)/(\d+) mode=(\S+) sup2=([0-9.]+) unsup2=([0-9.]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--mode", required=True, choices=["sup2", "both"])
    ap.add_argument("--total-iters", type=int, default=32000)
    ap.add_argument("--base-gamma", type=float, default=0.2)
    ap.add_argument("--unsup-weight", type=float, default=2.0)
    ap.add_argument("--start-ratio", type=float, default=0.5)
    ap.add_argument("--end-ratio", type=float, default=1.0)
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--tol", type=float, default=1e-8)
    args = ap.parse_args()

    rows = []
    with open(args.log, errors="replace") as handle:
        for line in handle:
            m = LINE.search(line)
            if m:
                rows.append((int(m.group(1)), int(m.group(2)), m.group(3),
                             float(m.group(4)), float(m.group(5))))

    T = args.total_iters
    if not rows:
        sys.exit("FAIL: no [PG weights] line found in " + args.log)

    failures = []

    # 1) expected set of logged iterations
    expected_iters = sorted({1} | {i for i in range(1, T + 1) if i % args.log_interval == 0} | {T})
    got_iters = [r[0] for r in rows]
    if got_iters != expected_iters:
        missing = sorted(set(expected_iters) - set(got_iters))[:5]
        extra = sorted(set(got_iters) - set(expected_iters))[:5]
        failures.append("iteration set mismatch: got %d lines, expected %d (missing %s, extra %s)"
                        % (len(got_iters), len(expected_iters), missing, extra))

    # 2) mode
    modes = {r[2] for r in rows}
    if modes != {args.mode}:
        failures.append("mode is %s, expected %s" % (sorted(modes), args.mode))

    # 3) per-line closed-form check
    worst = 0.0
    for it, total, mode, sup2, unsup2 in rows:
        if total != T:
            failures.append("iter %d logged /%d, expected /%d" % (it, total, T))
            break
        progress = (it - 1) / float(T - 1)
        ratio = args.start_ratio + (args.end_ratio - args.start_ratio) * progress
        gamma = args.base_gamma * ratio
        multiplier = gamma if args.mode == "both" else args.base_gamma
        exp_sup2, exp_unsup2 = gamma, args.unsup_weight * multiplier
        worst = max(worst, abs(sup2 - exp_sup2), abs(unsup2 - exp_unsup2))
        if abs(sup2 - exp_sup2) > args.tol or abs(unsup2 - exp_unsup2) > args.tol:
            failures.append("iter %d: sup2=%.9f (exp %.9f) unsup2=%.9f (exp %.9f)"
                            % (it, sup2, exp_sup2, unsup2, exp_unsup2))
            if len(failures) > 5:
                break

    # 4) endpoints and monotonicity
    exp_first = (args.base_gamma * args.start_ratio,
                 args.unsup_weight * (args.base_gamma * args.start_ratio
                                      if args.mode == "both" else args.base_gamma))
    exp_last = (args.base_gamma * args.end_ratio,
                args.unsup_weight * (args.base_gamma * args.end_ratio
                                     if args.mode == "both" else args.base_gamma))
    for label, got, exp in (("first", (rows[0][3], rows[0][4]), exp_first),
                            ("last", (rows[-1][3], rows[-1][4]), exp_last)):
        if abs(got[0] - exp[0]) > args.tol or abs(got[1] - exp[1]) > args.tol:
            failures.append("%s line sup2=%.9f unsup2=%.9f, expected %.9f/%.9f"
                            % (label, got[0], got[1], exp[0], exp[1]))
    sup2s = [r[3] for r in rows]
    if any(b < a - args.tol for a, b in zip(sup2s, sup2s[1:])):
        failures.append("sup2 is not non-decreasing")
    unsup2s = [r[4] for r in rows]
    if args.mode == "both":
        if any(b < a - args.tol for a, b in zip(unsup2s, unsup2s[1:])):
            failures.append("unsup2 is not non-decreasing in mode=both")
    elif max(unsup2s) - min(unsup2s) > args.tol:
        failures.append("unsup2 varies in mode=sup2 (should be constant %.9f)"
                        % (args.unsup_weight * args.base_gamma))

    print("lines=%d  iters %d..%d  mode=%s" % (len(rows), rows[0][0], rows[-1][0], rows[0][2]))
    print("first: sup2=%.9f unsup2=%.9f" % (rows[0][3], rows[0][4]))
    print("last : sup2=%.9f unsup2=%.9f" % (rows[-1][3], rows[-1][4]))
    print("max |deviation from closed form| = %.3e" % worst)
    if failures:
        print("\nFAIL (%d):" % len(failures))
        for f in failures:
            print("  - " + f)
        sys.exit(2)
    print("\nPASS: 全部 %d 行与闭式公式一致，端点与单调性均符合" % len(rows))


if __name__ == "__main__":
    main()
