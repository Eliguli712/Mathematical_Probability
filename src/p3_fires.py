"""Problem 3 — fit Pareto and exponential models to heavy-tailed fire sizes.

The script uses one internally consistent exponential model on [0, infinity):
its MLE is 1 / sample_mean, its density is lambda * exp(-lambda * a), and its
survival function is exp(-lambda * a).
"""
import json

import numpy as np
import matplotlib.pyplot as plt

from common import path, use_style, RED, BLUE, GRAY, GREEN

# ------------------------------------------------- the assignment script
np.random.seed(42)
alpha_true, m_true, n_samples = 1.5, 10.0, 5000
u = np.random.uniform(0, 1, n_samples)
fire_acres = m_true * (1 - u)**(-1 / alpha_true)          # inverse transform

m_hat = np.min(fire_acres)                                 # MLE of m  (2(d))
alpha_hat = n_samples / np.sum(np.log(fire_acres / m_hat)) # MLE of alpha (2(d))
lambda_hat = 1.0 / np.mean(fire_acres)                     # Exp(rate) MLE

print(f"m_hat      = {m_hat:.6f}   (true m = {m_true})")
print(f"alpha_hat  = {alpha_hat:.6f}   (true alpha = {alpha_true})")
print(f"lambda_hat = {lambda_hat:.6f}   (= 1 / sample mean {fire_acres.mean():.3f})")
print(f"sample median = {np.median(fire_acres):.3f}, sample max = {fire_acres.max():.1f}")

fig, ax = plt.subplots(figsize=(8, 5))
bins = np.logspace(np.log10(m_hat), np.log10(np.max(fire_acres)), 50)
ax.hist(fire_acres, bins=bins, density=True, alpha=0.4, color="gray", label="Synthetic Fire Data")
x = np.logspace(np.log10(m_hat), np.log10(np.max(fire_acres)), 500)
pareto_pdf = (alpha_hat * (m_hat**alpha_hat)) / (x**(alpha_hat + 1))
exp_pdf = lambda_hat * np.exp(-lambda_hat * x)
ax.plot(x, pareto_pdf, "r-", linewidth=2, label=rf"Pareto Fit ($\hat{{\alpha}}={alpha_hat:.2f}$)")
ax.plot(x, exp_pdf, "b--", linewidth=2, label=rf"Exponential Fit ($\hat{{\lambda}}={lambda_hat:.4f}$)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Burnt Acres (a)", fontsize=12); ax.set_ylabel("Probability Density f(a)", fontsize=12)
ax.set_title("Pareto vs. Exponential Fit to Synthetic Fire Data (Log-Log Scale)", fontsize=13)
ax.grid(True, which="both", ls="--", alpha=0.5); ax.legend(fontsize=11)
plt.tight_layout(); plt.savefig(path("fig_fire_orig.png"), dpi=200); plt.close()

# --------------------------------------- 3(c) tail-ratio diagnostic, s = 10
def ccdf(t):
    return np.mean(fire_acres > t)

thr = [10, 100, 1000]
R_emp, R_exp = [], []
for a in thr:
    upper = ccdf(10*a)
    R_emp.append(ccdf(a)/upper if upper > 0 else float("inf"))
    R_exp.append(float(np.exp(9*lambda_hat*a)))
R_par = float(10**alpha_hat)
print("thresholds a ->10a:", thr)
print("empirical ratios :", [f"{r:.2f}" for r in R_emp])
print(f"Pareto ratio     : 10**alpha_hat = {R_par:.2f} (same at every threshold)")
print("exponential      :", [f"{r:.3g}" for r in R_exp])
print("counts above a, 10a:", [(a, int((fire_acres > a).sum()), int((fire_acres > 10*a).sum())) for a in thr])

# --------------------------------------- 3(a,b,d) catastrophic-fire prediction
A, N_TOT = 1e5, 23*70_000
P_par = float((m_hat/A)**alpha_hat)
P_exp = float(np.exp(-lambda_hat*A))
P_emp = float(ccdf(A))
P_true = float((m_true/A)**alpha_true)
print(f"P(> 100,000 acres): Pareto {P_par:.3e} | exponential {P_exp:.3e} | empirical {P_emp:.3e} | truth {P_true:.3e}")
print(f"expected count over {N_TOT:,} fires: Pareto {P_par*N_TOT:.2f} | exponential {P_exp*N_TOT:.2f} "
      f"| empirical {P_emp*N_TOT:.2f} | truth {P_true*N_TOT:.2f}")
assert P_exp == 0.0 and P_emp == 0.0          # both underflow / see nothing that large

fire_summary = dict(
    m_hat=float(m_hat),
    alpha_hat=float(alpha_hat),
    lambda_hat=float(lambda_hat),
    exponential_model="unshifted",
    mean=float(fire_acres.mean()),
    mx=float(fire_acres.max()),
    med=float(np.median(fire_acres)),
    R_emp=[float(r) for r in R_emp],
    R_par=R_par,
    R_exp=R_exp,
    P_par=P_par,
    P_exp=P_exp,
    P_emp=P_emp,
    P_true=P_true,
    Ntot=N_TOT,
)
with path("fire.json").open("w", encoding="utf-8") as handle:
    json.dump(fire_summary, handle, indent=2)
np.save(path("fire_acres.npy"), fire_acres)

# ----------------------------------------------------------------- figures
use_style()

fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.1))
ax = axs[0]
ax.hist(fire_acres, bins=bins, density=True, alpha=0.45, color=GRAY, label="synthetic data (n = 5000)")
gg = np.logspace(np.log10(m_hat), np.log10(fire_acres.max()), 500)
ax.plot(gg, alpha_hat*m_hat**alpha_hat/gg**(alpha_hat+1), color=RED, lw=1.8,
        label=rf"Pareto fit $\hat\alpha={alpha_hat:.3f}$")
ax.plot(gg, lambda_hat*np.exp(-lambda_hat*gg), "--", color=BLUE, lw=1.6,
        label=rf"Exponential fit $\hat\lambda={lambda_hat:.4f}$")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-10, 1)
ax.set_xlabel("burnt acres $a$"); ax.set_ylabel("density $f(a)$")
ax.set_title(r"pdf: Pareto is a line of slope $-(\hat\alpha+1)$")
ax.legend(fontsize=7, frameon=False, loc="lower left"); ax.grid(alpha=0.3, which="both")

ax = axs[1]
xs = np.sort(fire_acres); cc = 1 - np.arange(len(xs))/len(xs)
ax.loglog(xs, cc, color=GRAY, lw=1.8, label="empirical CCDF")
g2 = np.logspace(np.log10(m_hat), 5.2, 300)
ax.loglog(g2, (m_hat/g2)**alpha_hat, color=RED, lw=1.5, label=rf"Pareto $\hat\alpha={alpha_hat:.3f}$")
ax.loglog(g2, np.exp(-lambda_hat*g2), "--", color=BLUE, lw=1.5,
          label=rf"Exponential $\hat\lambda={lambda_hat:.4f}$")
ax.axvline(A, color="k", ls=":", lw=1)
ax.annotate("100,000 acres", (A, 3e-7), xytext=(-4, 0), textcoords="offset points",
            rotation=90, fontsize=7, ha="right", va="bottom")
ax.set_ylim(1e-8, 1.4); ax.set_xlim(8, 3e5)
ax.set_xlabel("acres $a$"); ax.set_ylabel(r"$P(\tilde a>a)$")
ax.set_title("tail: only the power law reaches it")
ax.legend(fontsize=7, frameon=False, loc="lower left"); ax.grid(alpha=0.3, which="both")
plt.tight_layout(); plt.savefig(path("fig3.png"), dpi=220); plt.close()

fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.8))
ax = axs[0]
i = np.arange(3); w = 0.35
ax.bar(i-w/2, [R_emp[0], R_emp[1], np.nan], w, color=GRAY, label="empirical")
ax.bar(i+w/2, [R_par]*3, w, color=RED, label=rf"Pareto $10^{{\hat\alpha}}={R_par:.1f}$")
for k, v in enumerate(R_emp[:2]):
    ax.text(k-w/2, v+2, f"{v:.1f}", ha="center", fontsize=7.5)
ax.text(2-w/2, 3, "undefined\n(3 fires above 1000,\n0 above $10^4$)", ha="center", fontsize=6.5)
for k, v in enumerate(R_exp):
    ax.text(k, 74, f"exp: {v:.2g}", ha="center", fontsize=6.5, color=BLUE)
ax.set_ylim(0, 82); ax.set_xticks(i)
ax.set_xticklabels(["10→100", "100→1000", "1000→10⁴"], fontsize=8)
ax.set_ylabel(r"$P(\tilde a>a)/P(\tilde a>10a)$")
ax.set_title("tail-ratio diagnostic: roughly constant")
ax.legend(fontsize=7, frameon=False, loc="upper left"); ax.grid(alpha=0.3, axis="y")

ax = axs[1]
vals = [P_emp*N_TOT, P_exp*N_TOT, P_par*N_TOT, P_true*N_TOT]
bars = ax.bar(["Empirical", "Exponential", "Pareto", "Truth"],
              [max(v, 1e-3) for v in vals], color=[GRAY, BLUE, RED, GREEN], width=0.6)
for r, v in zip(bars, vals):
    ax.text(r.get_x()+r.get_width()/2, r.get_height()*1.2, "0" if v == 0 else f"{v:.2f}",
            ha="center", fontsize=8)
ax.set_yscale("log"); ax.set_ylim(1e-3, 20)
ax.set_ylabel("expected # fires > 100,000 acres")
ax.set_title("prediction: 23 yr × 70,000 fires/yr")
plt.setp(ax.get_xticklabels(), fontsize=8); ax.grid(alpha=0.3, axis="y", which="both")
plt.tight_layout(); plt.savefig(path("fig4.png"), dpi=220); plt.close()
print("p3_fires.py done")
