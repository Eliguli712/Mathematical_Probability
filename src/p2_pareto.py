"""Problem 2 — Pareto pdf and tail, compared with the exponential.

pdf   f(a) = alpha * m**alpha / a**(alpha+1)   for a >= m
tail  P(a~ > a) = (m/a)**alpha
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from common import path, use_style

use_style()
cols = {1: "#57068C", 2: "#8E44AD", 4: "#2471A3"}

# where the Pareto pdf crosses the Exp(1) pdf, for the discussion in the report
for alpha in (1, 2, 4):
    def log_density_ratio(value):
        return np.log(alpha) - (alpha + 1) * np.log(value) + value

    xs = np.linspace(1.0001, 40, 100_000)
    values = log_density_ratio(xs)
    brackets = np.where(np.diff(np.sign(values)))[0]
    roots = [
        round(brentq(log_density_ratio, xs[i], xs[i + 1]), 3)
        for i in brackets
    ]
    print(
        f"alpha = {alpha}: pdf(1) = {alpha} vs exp(-1) = {np.exp(-1):.4f}; "
        f"crossings at {roots or 'none (Pareto above throughout)'}"
    )
    print(
        f"           at a = 20: pareto {alpha / 20.0 ** (alpha + 1):.3e}, "
        f"exponential {np.exp(-20):.3e}"
    )

x = np.linspace(0, 20, 4001)
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.3))

ax = axs[0]
for alpha in (1, 2, 4):
    xp = x[x >= 1]
    ax.plot(
        xp,
        alpha / xp ** (alpha + 1),
        color=cols[alpha],
        lw=1.6,
        label=rf"Pareto $m=1,\ \alpha={alpha}$",
    )
ax.plot(x, np.exp(-x), "k--", lw=1.4, label=r"Exponential $\lambda=1$")
ax.axvline(1, color="gray", ls=":", lw=0.9)
ax.text(1.3, 2e-9, "Pareto pdf = 0\nfor $a<m=1$", fontsize=7, color="gray")
ax.set_yscale("log"); ax.set_ylim(1e-9, 10); ax.set_xlim(0, 20)
ax.set_xlabel("$a$"); ax.set_ylabel("pdf  (log scale)")
ax.set_title("2(b)  pdfs, logarithmic vertical axis")
ax.legend(fontsize=7.5, frameon=False, loc="upper right"); ax.grid(alpha=0.25, which="both")

ax = axs[1]
xl = np.logspace(0, 2, 400)
for alpha in (1, 2, 4):
    ax.loglog(
        xl,
        xl ** (-alpha),
        color=cols[alpha],
        lw=1.6,
        label=rf"$P(\widetilde A>a)=a^{{-{alpha}}}$",
    )
ax.loglog(xl, np.exp(-xl), "k--", lw=1.4, label=r"$e^{-a}$")
ax.set_ylim(1e-12, 1.5); ax.set_xlabel("$a$ (log)"); ax.set_ylabel(r"tail $P(\tilde a>a)$ (log)")
ax.set_title("Tails on log-log axes")
ax.legend(fontsize=7.5, frameon=False); ax.grid(alpha=0.25, which="both")

plt.tight_layout(); plt.savefig(path("fig2.png"), dpi=220); plt.close()
print("p2_pareto.py done")
