"""Problem 4 — KL divergence.

  * closed form for two multivariate Gaussians, checked against Monte Carlo
  * 4(a) maximum likelihood == minimizing D(p_X || p_theta), checked to machine precision
  * 4(b) second-order expansion == a weighted sum of squares, error checked to be O(eps^3)
"""
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

from common import path, use_style, RED, BLUE, GRAY, GREEN


# ---------------------------------------- Gaussian KL: closed form vs Monte Carlo
def kl_gaussian(m0, S0, m1, S1):
    """Return D(N(m0,S0) || N(m1,S1)) using stable linear solves."""

    d = len(m0)
    dm = m1 - m0
    sign0, logdet0 = np.linalg.slogdet(S0)
    sign1, logdet1 = np.linalg.slogdet(S1)
    if sign0 <= 0 or sign1 <= 0:
        raise ValueError("Covariance matrices must be positive definite.")
    trace_term = np.trace(np.linalg.solve(S1, S0))
    mean_term = dm @ np.linalg.solve(S1, dm)
    return 0.5 * (logdet1 - logdet0 - d + trace_term + mean_term)


def log_gauss_pdf(X, m, S):
    """Evaluate a multivariate Gaussian log density for row-wise samples."""

    d = X.shape[1]
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0:
        raise ValueError("Covariance matrix must be positive definite.")
    Z = X - m
    quadratic = np.einsum("ij,ji->i", Z, np.linalg.solve(S, Z.T))
    return -0.5 * (d * np.log(2 * np.pi) + logdet + quadratic)


np.random.seed(0)
d = 4
A = np.random.randn(d, d); S0 = A @ A.T + d*np.eye(d)
B = np.random.randn(d, d); S1 = B @ B.T + d*np.eye(d)
mu0, mu1 = np.random.randn(d), np.random.randn(d)

kc = kl_gaussian(mu0, S0, mu1, S1)
N_MC = 2_000_000
X = mu0 + np.random.randn(N_MC, d) @ np.linalg.cholesky(S0).T
diff = log_gauss_pdf(X, mu0, S0) - log_gauss_pdf(X, mu1, S1)
mc, se3 = diff.mean(), 3*diff.std()/np.sqrt(N_MC)
rev = kl_gaussian(mu1, S1, mu0, S0)
print(f"Gaussian KL: closed form {kc:.6f} nats, Monte Carlo {mc:.6f} +/- {se3:.6f} (3 s.e.)")
print(f"reverse direction {rev:.6f} -> KL is not symmetric")
print(f"self-divergence check: {kl_gaussian(mu0, S0, mu0, S0):.2e}")
assert abs(kc - mc) < se3
assert abs(kl_gaussian(mu0, S0, mu0, S0)) < 1e-12
with path("kl.json").open("w", encoding="utf-8") as handle:
    json.dump(
        dict(kc=float(kc), mc=float(mc), se3=float(se3), rev=float(rev)),
        handle,
        indent=2,
    )

use_style()
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9))
t = np.linspace(-6, 8, 800)
norm = lambda t, m, s: np.exp(-(t-m)**2/(2*s**2))/(s*np.sqrt(2*np.pi))
one = lambda m, sd: (np.array([m]), np.array([[sd**2]]))
fwd1 = kl_gaussian(*one(0, 1), *one(1, 2))
rev1 = kl_gaussian(*one(1, 2), *one(0, 1))
print(f"1-D example: D(N(0,1)||N(1,2^2)) = {fwd1:.4f}, reverse {rev1:.4f}")
ax = axs[0]
ax.plot(t, norm(t, 0, 1), color=RED, lw=1.6, label=r"$p=\mathcal{N}(0,1)$")
ax.plot(t, norm(t, 1, 2), color=BLUE, lw=1.6, label=r"$q=\mathcal{N}(1,2^2)$")
ax.fill_between(t, 0, norm(t, 0, 1), color=RED, alpha=0.12)
ax.set_title(rf"$D(p\|q)={fwd1:.3f}$ vs $D(q\|p)={rev1:.3f}$")
ax.set_xlabel("$x$"); ax.set_ylabel("density")
ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.25)
ax = axs[1]
s = np.linspace(0.2, 4, 400)
ax.plot(s, 0.5*(np.log(s**2) - 1 + 1/s**2), color=RED, lw=1.6,
        label=r"$D(\mathcal{N}(0,1)\,\|\,\mathcal{N}(0,s^2))$")
ax.plot(s, 0.5*(np.log(1/s**2) - 1 + s**2), color=BLUE, lw=1.6,
        label=r"$D(\mathcal{N}(0,s^2)\,\|\,\mathcal{N}(0,1))$")
ax.axhline(0, color="k", lw=0.6); ax.axvline(1, color="gray", ls=":", lw=1)
ax.set_ylim(-0.1, 4); ax.set_xlabel("$s$"); ax.set_ylabel("KL divergence (nats)")
ax.set_title("Nonnegative, zero only at $p=q$, asymmetric")
ax.legend(fontsize=7.5, frameon=False); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(path("fig5.png"), dpi=220); plt.close()

# ------------------------------- 4(a) MLE == argmin KL, on Binomial(K, theta)
rng = np.random.default_rng(11)
K, n, theta_true = 6, 800, 0.40
x = rng.binomial(K, theta_true, n)
a = np.arange(K+1)
pX = np.bincount(x, minlength=K+1) / n                    # empirical pmf
assert (pX > 0).all(), "the weighted-least-squares form needs full support"
H = -np.sum(pX*np.log(pX))                                # empirical entropy

pmod = lambda th: comb(K, a)*th**a*(1-th)**(K-a)
th = np.linspace(0.20, 0.62, 601)
avg_ll = np.array([np.sum(pX*np.log(pmod(t))) for t in th])       # (1/n) ln L
D = np.array([np.sum(pX*np.log(pX/pmod(t))) for t in th])         # D(p_X || p_theta)
th_ml, th_kl, closed = th[np.argmax(avg_ll)], th[np.argmin(D)], x.mean()/K
gap = np.max(np.abs(avg_ll + D + H))                              # identity: (1/n)lnL + D + H = 0
print(f"4(a): argmax (1/n)lnL = {th_ml:.4f}, argmin KL = {th_kl:.4f}, closed-form xbar/K = {closed:.4f}")
print(f"      max |(1/n)lnL + D + H| over the grid = {gap:.2e}   (H = {H:.4f} nats)")
assert th_ml == th_kl and gap < 1e-12

fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9))
ax = axs[0]
ax.plot(th, avg_ll, color=BLUE, lw=1.8, label=r"$\frac{1}{n}\ln L(\theta)$")
ax.plot(th, -D - H, "--", color=RED, lw=1.6, label=r"$-D(p_X\|p_\theta) - H(p_X)$")
ax.axvline(th_ml, color="k", ls=":", lw=1)
ax.text(th_ml+0.012, avg_ll.max()-0.02, rf"$\hat\theta={closed:.4f}$", fontsize=8)
ax.set_xlabel(r"$\theta$"); ax.set_ylabel("nats"); ax.set_title("The two curves coincide exactly")
ax.legend(fontsize=8, frameon=False, loc="lower left"); ax.grid(alpha=0.25)
ax = axs[1]
ax.plot(th, D, color=RED, lw=1.8, label=r"$D(p_X\|p_\theta)$")
ax.plot(th, [0.5*np.sum((pmod(t)-pX)**2/pX) for t in th], "--", color=GREEN, lw=1.5,
        label=r"$\frac{1}{2}\sum_a (p_\theta-p_X)^2/p_X(a)$")
ax.axvline(th_ml, color="k", ls=":", lw=1)
ax.set_ylim(0, 0.35); ax.set_xlabel(r"$\theta$"); ax.set_ylabel("nats")
ax.set_title("Quadratic approximation")
ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(path("fig6.png"), dpi=220); plt.close()

# ------------------------------- 4(b) order of the quadratic approximation
p0 = np.array([0.35, 0.25, 0.2, 0.12, 0.08])
dvec = np.array([1.0, -0.4, -0.3, 0.2, -0.5])
dvec = dvec - dvec.sum()/len(dvec)                 # keep the perturbed pmf normalized
eps = np.logspace(-4, -1.3, 40)
exactD = np.array([np.sum(p0*np.log(p0/(p0 + e*dvec))) for e in eps])
approx = np.array([0.5*np.sum((e*dvec)**2/p0) for e in eps])
rel = np.abs(approx - exactD)/exactD
slope = np.polyfit(np.log(eps), np.log(rel), 1)[0]
print(f"4(b): relative error at eps=1e-3 is {rel[np.argmin(abs(eps-1e-3))]:.3e}; "
      f"log-log slope of relative error vs eps = {slope:.3f}  (1 means absolute error is O(eps^3))")
assert 0.9 < slope < 1.2

fig, ax = plt.subplots(figsize=(4.6, 2.8))
ax.loglog(eps, exactD, color=RED, lw=1.8, label=r"exact $D(p_X\|p_\theta)$")
ax.loglog(eps, approx, "--", color=GREEN, lw=1.5, label=r"$\frac{1}{2}\sum(p_\theta-p_X)^2/p_X$")
ax.loglog(eps, rel, ":", color=GRAY, lw=1.6, label="relative error (slope 1)")
ax.set_xlabel(r"perturbation size $\epsilon$"); ax.set_ylabel("nats  /  relative error")
ax.set_title(r"Error is $O(\epsilon^3)$, i.e. relative $O(\epsilon)$", fontsize=9.5)
ax.legend(fontsize=7.5, frameon=False, loc="lower right"); ax.grid(alpha=0.3, which="both")
plt.tight_layout(); plt.savefig(path("fig7.png"), dpi=220); plt.close()

with path("kl2.json").open("w", encoding="utf-8") as handle:
    json.dump(
        dict(
            th_ml=float(th_ml),
            th_kl=float(th_kl),
            closed=float(closed),
            H=float(H),
            gap=float(gap),
            minD=float(D.min()),
            n=n,
            K=K,
            theta_true=theta_true,
            rel1e3=float(rel[np.argmin(abs(eps - 1e-3))]),
            slope=float(slope),
        ),
        handle,
        indent=2,
    )
print("p4_kl.py done")
