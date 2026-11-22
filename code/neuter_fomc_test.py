"""Raw against neutered scores of FOMC statements: the look-ahead test (19 Nov 2026).

No language model is used here. The script reads the scores made by neuter_fomc.py and the 30-minute rate
changes of USMPD. For the raw and for the neutered scores it reports:
  1. MOVE: the correlation of the model's guess of the 2-year yield move with the true move in the window.
  2. The canonical correlations between the three scores (TARGET, PATH, BALANCE) and the 12 rate series.
  3. r: the number of rate factors that are correlated with the scores (Ahn and Perez 2010 with candidate
     variables, as in Ahn, Dieckmann and Perez 2018). With three scores the criteria can reach 2.
  4. The R squared of the first three principal components of the rate changes on the three scores.
  5. A placebo outcome that the text cannot know: the change of the 2-year yield over the five trading days
     after the event day (daily yields of Gurkaynak, Sack and Wright, feds200628.csv).
The gap raw minus neutered is the estimate of the look-ahead leak (an upper bound, Proposition 1 of the paper).
Inference: a paired bootstrap over events (the same events for both versions). P is one-sided: gap > 0.
The same numbers are shown for the events after the knowledge cutoff of the model, where a leak is impossible.

Notation: r is the number of factors. k is not used.
Usage, from inside code/:
  python3 neuter_fomc_test.py --selftest
  python3 neuter_fomc_test.py --run run_openai_gpt-5-nano-2025-08-07 --cutoff 2024-05-30
Output: a text file next to the scores (leak_test.txt) and the same text on the screen.
"""
import argparse
import os

import numpy as np
import pandas as pd

import ap_tools_v2 as ap
import paths
from review_checks import FUT6, UST6, mode, scaled, shares, std, usmpd

ROOT = os.path.join(paths.DATA, "fomc_statements")
S = ["target", "path", "balance"]


def r2(y, X):
    X = np.column_stack([np.ones(len(X)), X])
    e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return 1 - e.var() / y.var()


def stats(Y, Z, move, y2, after):
    """Y rates (T x 12, standardized), Z three scores, move the model's guess, y2 the true 2-year move, after the placebo."""
    pcs = np.linalg.svd(Y - Y.mean(0), full_matrices=False)[0][:, :3]
    cc = ap.cca(Y, Z)
    corr = lambda a, b: np.corrcoef(a, b)[0, 1] if np.std(a) > 0 and np.std(b) > 0 else 0.0
    return np.array([corr(move, y2), (cc ** 2).sum(), cc[0], cc[1], cc[2], r2(pcs[:, 0], Z), r2(pcs[:, 1], Z), r2(pcs[:, 2], Z),
                     corr(move, after), r2(after, Z)])


NAMES = ["corr(MOVE, true 2-year move)", "sum of squared canonical correlations", "canonical correlation 1", "canonical correlation 2",
         "canonical correlation 3", "R2 of rate component 1", "R2 of rate component 2", "R2 of rate component 3",
         "placebo: corr(MOVE, next 5 days)", "placebo: R2 of the next 5 days"]


def report(d, label, n_boot, rng):
    Y = std(d[FUT6 + UST6].values)
    y2, after = d["UST2Y"].values, d["after5"].values
    Zr, Zn = d[[s + "_raw" for s in S]].values, d[[s + "_neutered" for s in S]].values
    obs_r, obs_n = stats(Y, Zr, d["move_raw"].values, y2, after), stats(Y, Zn, d["move_neutered"].values, y2, after)
    gaps = []
    for _ in range(n_boot):
        i = rng.integers(0, len(d), len(d))
        gaps.append(stats(Y[i], Zr[i], d["move_raw"].values[i], y2[i], after[i]) - stats(Y[i], Zn[i], d["move_neutered"].values[i], y2[i], after[i]))
    gaps = np.array(gaps)
    lines = [f"{label}: {len(d)} events", f"   {'statistic':40s} {'raw':>7s} {'neutered':>9s} {'gap':>7s}   90 percent interval of the gap     P(gap <= 0)"]
    for j, name in enumerate(NAMES):
        lo, hi = np.quantile(gaps[:, j], [0.05, 0.95])
        p = (1 + (gaps[:, j] <= 0).sum()) / (1 + n_boot)
        lines.append(f"   {name:40s} {obs_r[j]:7.3f} {obs_n[j]:9.3f} {obs_r[j] - obs_n[j]:+7.3f}   [{lo:+.3f}, {hi:+.3f}]                 {p:.3f}")
    W = scaled(d[FUT6 + UST6].values)
    for v, Z in [("raw", Zr), ("neutered", Zn)]:
        if len(d) >= 80 and all(np.std(Z, 0) > 0):
            res = ap.count_r(W, std(Z), 300, 0, np.random.default_rng(1))
            lines.append(f"   r with the {v} scores: BIC2 {{{shares(res['bic2'])}}}   sequential {{{shares(res['seq'])}}}   BIC3 {{{shares(res['bic3'])}}}")
        else:
            lines.append(f"   r with the {v} scores: not computed (fewer than 80 events, or a score that never moves)")
    return lines


def load(run, cutoff):
    sc = pd.read_csv(os.path.join(ROOT, run, "scores.csv"))
    u = usmpd("Statements")
    u["date"] = u["Date"].dt.strftime("%Y-%m-%d")
    g = pd.read_csv(os.path.join(paths.FED, "feds200628.csv"), skiprows=9, usecols=["Date", "SVENY02"], na_values=["NA"]).dropna()
    g["Date"] = pd.to_datetime(g["Date"])
    y = g.set_index("Date")["SVENY02"]
    u["after5"] = [(y.shift(-5).get(t, np.nan) - y.get(t, np.nan)) for t in u["Date"]]
    d = sc.merge(u[["date", "after5"] + FUT6 + UST6], on="date").dropna()
    return d[d.date <= cutoff], d[d.date > cutoff]


def selftest():
    rng = np.random.default_rng(0)
    u = usmpd("Statements")
    u["date"] = u["Date"].dt.strftime("%Y-%m-%d")
    d = u[["date"] + FUT6 + UST6].dropna().copy()
    Y = std(d[FUT6 + UST6].values)
    pcs = np.linalg.svd(Y, full_matrices=False)[0][:, :3] * np.sqrt(len(Y))
    d["after5"] = rng.standard_normal(len(d))
    for j, s in enumerate(S):
        d[s + "_neutered"] = np.round(0.6 * pcs[:, j] + rng.standard_normal(len(d)))
        d[s + "_raw"] = np.round(0.6 * pcs[:, j] + 0.6 * pcs[:, j] + 0.7 * rng.standard_normal(len(d)))     # a leak in every score
    d["move_neutered"] = np.round(2 * rng.standard_normal(len(d)))
    d["move_raw"] = np.round(40 * d["UST2Y"].values + 2 * rng.standard_normal(len(d)))                         # the model remembers the move
    lines = report(d, "selftest with a built-in leak", 199, rng)
    print("\n".join(lines))
    gap_line = [l for l in lines if l.strip().startswith("corr(MOVE, true")][0]
    assert float(gap_line.split()[-1]) < 0.05, "the built-in leak was not found"
    print("selftest passed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run")
    p.add_argument("--cutoff", default="2024-05-30", help="knowledge cutoff of the model (events after it cannot leak)")
    p.add_argument("--boot", type=int, default=999)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return selftest()
    rng = np.random.default_rng(20261119)
    before, after = load(a.run, a.cutoff)
    lines = [f"Raw against neutered scores. Run: {a.run}. Knowledge cutoff used: {a.cutoff}. Bootstrap draws: {a.boot}.", ""]
    lines += report(before, "Events up to the cutoff (a leak is possible)", a.boot, rng) + [""]
    if len(after) >= 10:
        lines += report(after, "Events after the cutoff (no leak is possible)", a.boot, rng)
    else:
        lines.append(f"Events after the cutoff: {len(after)}. Too few for a table.")
    with open(os.path.join(ROOT, a.run, "leak_test.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
