"""Check of adp_tools.py on simulated data shaped like our sector panel (10 Nov 2026).
n = 9 responses, T = 167 (and 56), k = 3 instruments. Two designs:
  (i)  m = 2, r = 2: a market factor and a rate factor, both seen by the instruments.
  (ii) m = 3, r = 2: the same plus a missing factor that the instruments do not see.
Instruments: z1 = market + noise, z2 = rate + noise, z3 = rate + noise. Idiosyncratic errors: t(5).
Reported: the share of replications in which the most frequent value over 100 specifications equals the truth.
Usage: python3 adp_simulation.py
"""
import numpy as np

from adp_tools import estimate_m, estimate_r, mode


def simulate(T, missing, rng):
    n = 9
    f = rng.standard_normal((T, 3))
    B = np.column_stack([rng.normal(1.0, 0.3, n), rng.normal(0, 0.6, n), rng.normal(0, 0.6, n)])
    if not missing:
        B[:, 2] = 0
    U = rng.standard_t(5, (T, n)) * 0.6
    Y = f @ B.T + U
    Z = np.column_stack([f[:, 0] + 0.5 * rng.standard_normal(T), f[:, 1] + 0.7 * rng.standard_normal(T),
                         f[:, 1] + 0.7 * rng.standard_normal(T)])
    return Y, Z


def main():
    rng = np.random.default_rng(20261110)
    for T in [167, 56]:
        for missing, m_true, r_true in [(False, 2, 2), (True, 3, 2)]:
            res = {"m_seq": 0, "m_bic": 0, "r_seq": 0, "r_bic": 0}
            R = 50
            for _ in range(R):
                Y, Z = simulate(T, missing, rng)
                ms, mb, _ = estimate_m(Y, 100, 3, 0.05, rng)
                rs, rb = estimate_r(Y, Z, 100, 3, 0.05, rng)
                res["m_seq"] += mode(ms)[0] == m_true
                res["m_bic"] += mode(mb)[0] == m_true
                res["r_seq"] += mode(rs)[0] == r_true
                res["r_bic"] += mode(rb)[0] == r_true
            print(f"T {T:3d}  true m {m_true}, r {r_true}:  " + ", ".join(f"{k} right {v / R:.0%}" for k, v in res.items()), flush=True)


if __name__ == "__main__":
    main()
