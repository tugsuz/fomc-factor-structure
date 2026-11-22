"""Matched-block permutation inference for the firm-side results (17 Nov 2026). Tools: panel_tools.py, panel_tools_v2.py.

Notation as in the lecture notes: r effects, r_o the true number, k regressors (k is never a number of factors; here k = 0),
q instruments, N stocks, T events.

The design, for one year:
  A block is one scheduled statement day with its own control days (one week before, one week after).
  A year has 8 blocks. The stocks are those with a return on every day of every block, so the panel is the same
  for all draws. The traits (instruments) and the scaling of each stock stay fixed.
  The statistic is computed from the N x 8 panel of the 8 statement days:
    cc1   the first canonical correlation between the 8 returns of a stock and its traits
    sum   the sum of all squared canonical correlations (the J statistic for r = 0, divided by N)
    r     the count by BIC2 (Ahn, Lee and Schmidt 2013, page 8), from the same canonical correlations
  A draw takes one day from each block at random (the statement day or one of its control days) and computes
  the same statistic. If statement days are like their control days, every draw is as likely as the observed set.
  P = (1 + number of draws at least as large as the observed value) / (1 + B).
Several years at once: the statistic is the mean over the years. A draw takes one day from each block of each year.
What is permuted: only which day of a block is used. What stays fixed: the stocks, the traits, the scaling.

sim   size and power at N = 435 stocks, 8 blocks of 3 days, q = 5 traits.
        null    all days alike (two effects tied to the traits, correlation 0.6)
        larger  the same effects move 1.25 times as much on statement days
        new     statement days have one more effect tied to the traits (correlation 0.4, half the size)
data  the tests on the TAQ panel for the instrument sets named in SETS (needs traits_v2.csv from panel_traits_v2.py).
Usage: python3 panel_perm.py sim     or     python3 panel_perm.py data
Output: results/published_check_2026-11-17/perm_sim.txt, perm_data.txt, perm_by_year.csv
"""
import os
import sys

import numpy as np
import pandas as pd

import panel_tools as pt
import panel_tools_v2 as p2
import paths

OUT = os.path.join(paths.RESULTS, "published_check_2026-11-17")
STATS = ["cc1", "sum", "r"]


def stats(Y, Qs, q):
    """The three statistics for the N x T panel Y. Qs is the orthonormal basis of the demeaned instruments."""
    N, T = Y.shape
    qy = np.linalg.qr(Y - Y.mean(0))[0]
    c2 = np.linalg.svd(qy.T @ Qs, compute_uv=False) ** 2
    pen = 0.75 * np.log(N) / np.log(5)                    # BIC2, as in panel_tools_v2.pick_v2 (written out for speed)
    tail = np.cumsum(c2[::-1])[::-1]
    best, count = 0.0, min(T, q)                         # the criterion is 0 at the largest value
    for r in range(min(T, q)):
        v = N * tail[r] - pen * (T - r) * (q - r)
        if v < best:
            best, count = v, r
    return np.sqrt(c2[0]), c2.sum(), count


def perm_year(R, blocks, S, n_draw, rng):
    """R: N x D returns of all block days. blocks: list of lists of column numbers, the statement day first.
    Returns the observed statistics (3) and the draws (n_draw x 3)."""
    Qs = np.linalg.qr(S - S.mean(0))[0]
    q = S.shape[1]
    obs = np.array(stats(R[:, [b[0] for b in blocks]], Qs, q))
    draws = np.array([stats(R[:, [b[rng.integers(len(b))] for b in blocks]], Qs, q) for _ in range(n_draw)])
    return obs, draws


def pval(obs, draws):
    return (1 + np.sum(draws >= obs - 1e-12)) / (1 + len(draws))


def sim(rng):
    N, q = 435, 5
    blocks = [[3 * j, 3 * j + 1, 3 * j + 2] for j in range(8)]

    def year(case, n_draw):
        S = rng.standard_normal((N, q))
        al = 0.6 * S[:, :2] + 0.8 * rng.standard_normal((N, 2))
        th = 0.4 * rng.standard_normal((2, 24))
        if case == "larger":
            th[:, ::3] *= 1.25
        Y = al @ th + rng.standard_t(5, (N, 24)) / np.sqrt(5 / 3)
        if case == "new":
            a3 = 0.4 * S[:, 2] + np.sqrt(1 - 0.16) * rng.standard_normal(N)
            Y[:, ::3] += np.outer(a3, 0.2 * rng.standard_normal(8))
        return perm_year(Y, blocks, S, n_draw, rng)

    lines = ["Share of simulated data sets with P at or below 0.05. Under 'null' this is the size of the test (it should be near 0.05).",
             "One year: 200 data sets with 199 draws. 21 years together: 30 data sets with 99 draws."]
    for case in ["null", "larger", "new"]:
        one = np.array([[pval(o[j], d[:, j]) for j in range(3)] for o, d in (year(case, 199) for _ in range(200))])
        pooled = []
        for _ in range(30):
            ys = [year(case, 99) for _ in range(21)]
            o = np.mean([y[0] for y in ys], axis=0)
            d = np.mean([y[1] for y in ys], axis=0)
            pooled.append([pval(o[j], d[:, j]) for j in range(3)])
        pooled = np.array(pooled)
        lines.append(f"{case:6s}  one year: " + "  ".join(f"{s} {np.mean(one[:, j] <= 0.05):.2f}" for j, s in enumerate(STATS)) +
                     "   | 21 years: " + "  ".join(f"{s} {np.mean(pooled[:, j] <= 0.05):.2f}" for j, s in enumerate(STATS)))
    return lines


# ---------------------------------------------------------------- the tests on the data
RISK = pt.RISK
SECT = pt.SECT
BAL = ["blev", "mlev", "bm", "cash", "stdebt", "intexp", "prof", "inv", "tang"]
PERIODS = [(2005, 2025), (2005, 2012), (2013, 2019), (2020, 2025)]


def net(t, cols, controls):
    """The columns of t, with the part explained by the controls (and a constant) taken out."""
    Z = np.column_stack([np.ones(len(t)), t[controls].values])
    R = t[cols].values
    return R - Z @ np.linalg.lstsq(Z, R, rcond=None)[0]


SETS = {
    "risk traits (5)": lambda t: t[RISK].values,
    "sector dummies (6)": lambda t: t[SECT].values,
    "rate betas, net of other risk traits (2)": lambda t: net(t, ["beta_y2", "beta_slope"], ["beta_mkt", "logcap", "vol"]),
    "balance sheet traits (9)": lambda t: t[BAL].values,
    "risk and balance sheet traits (14)": lambda t: t[RISK + BAL].values,
    "balance sheet, net of sector and risk traits (9)": lambda t: net(t, BAL, SECT + RISK),
}
for _c in BAL:
    SETS[f"{_c} alone, net of sector and risk traits (1)"] = (lambda c: lambda t: net(t, [c], SECT + RISK))(_c)


def year_blocks(d, st, A, B, year, tr, col="w30"):
    """The blocks of one year, the stocks with a return on every block day and with traits, the scaled returns."""
    s = [x for x in st if x[:4] == str(year)]
    ok = d.dropna(subset=[col]).groupby("date")["permno"].nunique()
    blocks = [[x] + [c for c in (A.get(x), B.get(x)) if c is not None and ok.get(c, 0) >= 300] for x in s]
    days = [x for b in blocks for x in b]
    pn, w = pt.panel(d, days, [col])
    t = tr[tr["year"] == year].set_index("permno")
    keep = [i for i, p in enumerate(pn) if p in t.index]
    t = t.loc[[pn[i] for i in keep]]
    pos = {x: i for i, x in enumerate(days)}
    return w[col][keep] / t["vol"].values[:, None], [[pos[x] for x in b] for b in blocks], t


def data(rng, n_draw=999):
    d = pt.stock_windows()
    st, A, B = pt.day_sets(d)
    tr = pd.read_csv(os.path.join(OUT, "traits_v2.csv"))
    for s in SECT:
        tr[s] = (tr["sector"] == s).astype(float)
    panels = {y: year_blocks(d, st, A, B, y, tr) for y in range(2005, 2026)}
    n = [len(v[2]) for v in panels.values()]
    lines = [f"Stocks per year: {min(n)} to {max(n)}. Blocks per year: {min(len(v[1]) for v in panels.values())} to {max(len(v[1]) for v in panels.values())}. "
             f"Draws: {n_draw}.", "For each set and sample: the mean over the years of the statistic on statement days, the mean over the draws, and P.",
             "cc1 = first canonical correlation. sum = sum of squared canonical correlations. r = count by BIC2.", ""]
    rows = []
    variants = [("all stocks", lambda t: np.ones(len(t), bool))]
    for name, build in SETS.items():
        for vname, mask in variants + ([("without financial firms and real estate", lambda t: ((t["fin"] + t["reit"]) == 0).values)]
                                       if name.startswith("balance sheet, net") else []):
            res = {}
            for y, (R, blocks, t) in panels.items():
                m = mask(t)
                S = build(t[m])
                S = S[:, S.std(0) > 0]
                res[y] = perm_year(R[m], blocks, S, n_draw, rng)
                for j, sname in enumerate(STATS):
                    rows.append({"set": name, "stocks": vname, "year": y, "stat": sname, "statement": res[y][0][j],
                                 "draw_mean": res[y][1][:, j].mean(), "P": pval(res[y][0][j], res[y][1][:, j])})
            lines.append(f"{name} | {vname}")
            for a, b in PERIODS:
                ys = [y for y in res if a <= y <= b]
                o = np.mean([res[y][0] for y in ys], axis=0)
                dr = np.mean([res[y][1] for y in ys], axis=0)
                yearly = sum(pval(res[y][0][0], res[y][1][:, 0]) <= 0.05 for y in ys)
                lines.append(f"   {a}-{b}: " + "   ".join(f"{s} {o[j]:.3f} / {dr[:, j].mean():.3f} / P {pval(o[j], dr[:, j]):.3f}" for j, s in enumerate(STATS)) +
                             f"   | years with P of cc1 at or below 0.05: {yearly} of {len(ys)}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "perm_by_year.csv"), index=False)
    return lines


if __name__ == "__main__":
    mode = sys.argv[1]
    rng = np.random.default_rng(20261122)
    if mode == "sim":
        lines = [__doc__.split("Usage")[0].strip(), ""] + sim(rng)
        with open(os.path.join(OUT, "perm_sim.txt"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(lines[-5:]))
    else:
        lines = [__doc__.split("Usage")[0].strip(), ""] + data(rng)
        with open(os.path.join(OUT, "perm_data.txt"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(l for l in lines[-200:] if "alone" not in l or True)[-9000:])
