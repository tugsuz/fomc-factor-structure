"""Simulations at our own sample sizes, before any use on data (17 Nov 2026). Tools: panel_tools.py.

Notation as in the lecture notes: r effects in estimation, r_o the true number, k regressors (k is not a number of
factors), q instruments, N units, T periods.

als0  Ahn, Lee and Schmidt with no regressor (k = 0). N = 440 stocks, T = 8 events, q = 5 traits as instruments.
      r_o effects have loadings correlated with the traits (correlation rho). One more effect is not correlated
      with the traits. The errors differ in size across units and have heavy tails (t with 5 degrees of freedom).
      Each unit is divided by a noisy measure of its own volatility, as we do with the data.
als1  The same with one regressor (k = 1). The regressor loads on the effects. The instruments are the T values
      of the regressor (q = 8), as on the lecture notes. True beta = 0.1.
bai   Bai (2009) at (N, T) = (350, 56) and (280, 167). Two factors. The regressor loads on the same loadings.
low   Bai (2009) with a regressor of the form trait_i * shock_t (a "low-rank" regressor), N = 350, T = 56.
        apart: trait and shock are independent of the loadings and factors. True beta = 0.1.
        close: the shock has correlation 0.9 with factor 1 and the trait 0.9 with its loadings. True beta = 0.1.
        same:  there is no regressor in the true model. Factor 1 is the true shock. We observe it with error,
               and the trait has correlation 0.6 with its loadings.
Usage: python3 panel_sim.py als0|als1|bai|low     Output: results/panel_effects_2026-11-17/sim_<name>.txt
"""
import os
import sys

import numpy as np

import panel_tools as pt


def noise(rng, N, T):
    sig = np.exp(0.4 * rng.standard_normal(N))
    e = rng.standard_t(5, (N, T)) / np.sqrt(5 / 3) * sig[:, None]
    return e, sig * np.exp(0.15 * rng.standard_normal(N))


def als0(rng, reps=150, N=440, T=8, q=5):
    lines = ["share of panels with the true r_o (mean estimate in brackets)"]
    for ro in [1, 2, 3]:
        for rho in [0.3, 0.6]:
            res = {m: [] for m in ["cu seq", "cu bic", "cu bic2", "gmm seq", "gmm bic2"]}
            for _ in range(reps):
                S = rng.standard_normal((N, q))
                G = np.linalg.qr(rng.standard_normal((q, ro)))[0]
                al = rho * S @ G + np.sqrt(1 - rho ** 2) * rng.standard_normal((N, ro))
                e, v = noise(rng, N, T)
                Y = (0.4 * al @ rng.standard_normal((ro, T)) + 0.4 * rng.standard_normal((N, 1)) @ rng.standard_normal((1, T)) + e) / v[:, None]
                a = pt.als_count(Y, None, S, "cu")
                b = pt.als_count(Y, None, S, "gmm", n_order=3, rng=rng)
                for m, x in zip(res, [a["seq"], a["bic"], a["bic2"], b["seq"], b["bic2"]]):
                    res[m].append(x)
            lines.append(f"r_o {ro} rho {rho}:  " + "   ".join(f"{m} {np.mean(np.array(x) == ro):.2f} ({np.mean(x):.2f})" for m, x in res.items()))
    return lines


def als1(rng, reps=150, N=440, T=8, beta=0.1):
    lines = ["r_o: share of panels with the true value. beta: mean, standard deviation over the panels. True beta 0.1."]
    for ro in [1, 2]:
        rr = {m: [] for m in ["cu seq", "cu bic", "cu bic2", "gmm bic2"]}
        bb = {m: [] for m in ["fixed effects", "cu at BIC2 r", "cu at r_o", "gmm at r_o", "cu at r_o + 1"]}
        for i in range(reps):
            al = rng.standard_normal((N, ro))
            phi = 0.5 * rng.standard_normal((ro, T))
            th = 0.7 * phi + 0.35 * rng.standard_normal((ro, T))
            e, v = noise(rng, N, T)
            x = al @ phi + rng.standard_normal((N, T))
            y = beta * x + al @ th + e
            x, y = x / v[:, None], y / v[:, None]
            X = x[:, :, None]
            a = pt.als_count(y, X, x, "cu")
            rr["cu seq"].append(a["seq"])
            rr["cu bic"].append(a["bic"])
            rr["cu bic2"].append(a["bic2"])
            bb["fixed effects"].append(pt.fe_ols(y, X)[0])
            bb["cu at BIC2 r"].append(a["beta"][min(a["bic2"], max(a["beta"]))][0])
            bb["cu at r_o"].append(a["beta"][ro][0])
            bb["cu at r_o + 1"].append(a["beta"][ro + 1][0])
            if i < 60:
                g = pt.als_count(y, X, x, "gmm", n_order=3, rng=rng)
                rr["gmm bic2"].append(g["bic2"])
                bb["gmm at r_o"].append(np.median([b[ro][0] for b in g["betas"]]))
        lines.append(f"r_o {ro}:  " + "   ".join(f"{m} {np.mean(np.array(x) == ro):.2f} ({np.mean(x):.2f})" for m, x in rr.items()))
        lines.append("        " + "   ".join(f"{m}: {np.mean(x):.3f}, {np.std(x):.3f}" for m, x in bb.items()))
    return lines


def bai(rng, reps=200, beta=0.1):
    lines = ["For each r used: mean of beta, standard deviation, and the share of 5 percent t-tests of the true beta that reject",
             "(first with the standard errors of the lecture notes, then with standard errors clustered by period). True beta 0.1, r_o = 2."]
    for N, T in [(350, 56), (280, 167)]:
        out = {r: [] for r in [0, 1, 2, 4, 6]}
        for _ in range(reps):
            lam, f, h = rng.standard_normal((N, 2)), rng.standard_normal((T, 2)), rng.standard_normal((T, 2))
            e, _ = noise(rng, N, T)
            x = 0.6 * lam @ (f + h).T + rng.standard_normal((N, T))
            y = beta * x + 0.6 * lam @ f.T + e
            for r in out:
                b, s1, s2, _, _ = pt.bai_ife(y, x[:, :, None], r)
                out[r].append((b[0], abs(b[0] - beta) / s1[0] > 1.96, abs(b[0] - beta) / s2[0] > 1.96))
        for r, v in out.items():
            v = np.array(v, float)
            lines.append(f"N {N} T {T} r {r}:  mean {v[:, 0].mean():.4f}  sd {v[:, 0].std():.4f}  reject {v[:, 1].mean():.2f} {v[:, 2].mean():.2f}")
    return lines


def low(rng, reps=200, N=350, T=56):
    lines = ["Regressor trait_i * shock_t. For each r used: mean and standard deviation of beta over the panels."]
    for case in ["apart", "close", "same"]:
        out = {r: [] for r in [0, 1, 2, 3]}
        for _ in range(reps):
            lam, f = rng.standard_normal((N, 2)), rng.standard_normal((T, 2))
            e, _ = noise(rng, N, T)
            c = {"apart": 0.0, "close": 0.9, "same": 0.6}[case]
            w = c * lam[:, 0] + np.sqrt(1 - c ** 2) * rng.standard_normal(N)
            if case == "same":
                v = f[:, 0] + 0.5 * rng.standard_normal(T)
                y = 0.6 * lam @ f.T + e
            else:
                cv = 0.0 if case == "apart" else 0.9
                v = cv * f[:, 0] + np.sqrt(1 - cv ** 2) * rng.standard_normal(T)
                y = 0.1 * np.outer(w, v) + 0.6 * lam @ f.T + e
            X = np.outer(w, v)[:, :, None]
            for r in out:
                out[r].append(pt.bai_ife(y, X, r)[0][0])
        lines.append(f"{case:5s}:  " + "   ".join(f"r {r}: {np.mean(v):.3f}, {np.std(v):.3f}" for r, v in out.items()))
    return lines


if __name__ == "__main__":
    name = sys.argv[1]
    rng = np.random.default_rng(20261117)
    lines = [__doc__.split("Usage")[0].strip(), "", name] + {"als0": als0, "als1": als1, "bai": bai, "low": low}[name](rng)
    with open(os.path.join(pt.OUT, f"sim_{name}.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[lines.index(name):]))
