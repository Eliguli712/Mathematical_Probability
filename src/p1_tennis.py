"""Problem 1 — pmf of the total number of games in an Alcaraz-Sinner match.

Part A reproduces the assignment code verbatim (same RNG call sequence, seed 42).
Part B adds an exact Markov-chain computation of the same model, used to validate
the Monte Carlo, plus two sensitivity variants and the report figures.
"""
import json
from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt

from common import path, use_style, RED, C3, C4, C5

# ----------------------------------------------------------------- data
matches_scores = [
    [(1, 6), (4, 6), (7, 6), (3, 6)],
    [(6, 3), (6, 7), (6, 7), (7, 5), (6, 3)],
    [(2, 6), (6, 3), (3, 6), (6, 4), (6, 3)],
    [(4, 6), (6, 7), (6, 4), (7, 6), (7, 6)],
    [(6, 4), (4, 6), (4, 6), (4, 6)],
    [(6, 2), (3, 6), (6, 1), (6, 4)],
]
total_games = [sum(g1 + g2 for g1, g2 in m) for m in matches_scores]
n_sets = [len(m) for m in matches_scores]

alc_won = [76, 104, 93, 117, 77, 65]
alc_tot = [121, 169, 157, 194, 121, 89]
sin_won = [101, 118, 83, 116, 81, 65]
sin_tot = [143, 213, 135, 191, 117, 112]

p_alc_serve = sum(alc_won) / sum(alc_tot)
p_sin_serve = sum(sin_won) / sum(sin_tot)

print("Total games in 6 matches:", total_games)
print(f"Overall Alcaraz win prob on serve: {p_alc_serve:.4f} ({sum(alc_won)}/{sum(alc_tot)})")
print(f"Overall Sinner win prob on serve: {p_sin_serve:.4f} ({sum(sin_won)}/{sum(sin_tot)})")


# ------------------------------------------------- simulation (as supplied)
def simulate_game(p_server):
    """One standard game: first to 4 points, margin >= 2. True iff server holds."""
    pts_server = pts_returner = 0
    while True:
        if np.random.rand() < p_server:
            pts_server += 1
        else:
            pts_returner += 1
        if pts_server >= 4 and pts_server - pts_returner >= 2:
            return True
        if pts_returner >= 4 and pts_returner - pts_server >= 2:
            return False


def simulate_tiebreak(p_server_A, p_server_B, server_starts_A):
    """7-point tie break, margin >= 2. Serve pattern A | BB | AA | BB ...  True iff A wins."""
    pts_A = pts_B = pts_played = 0
    while True:
        if pts_played == 0:
            current_server_is_A = server_starts_A
        else:
            block = (pts_played + 1) // 2
            current_server_is_A = (not server_starts_A) if block % 2 == 1 else server_starts_A
        p_win_A = p_server_A if current_server_is_A else (1 - p_server_B)
        if np.random.rand() < p_win_A:
            pts_A += 1
        else:
            pts_B += 1
        pts_played += 1
        if pts_A >= 7 and pts_A - pts_B >= 2:
            return True
        if pts_B >= 7 and pts_B - pts_A >= 2:
            return False


def simulate_set(p_A, p_B, server_starts_A):
    """Returns (games played, A won the set, who serves first in the next set)."""
    games_A = games_B = 0
    current_server_is_A = server_starts_A
    while True:
        if games_A == 6 and games_B == 6:
            if simulate_tiebreak(p_A, p_B, current_server_is_A):
                games_A += 1
            else:
                games_B += 1
            # whoever served first in the tie break receives first next set
            return games_A + games_B, games_A > games_B, not current_server_is_A
        server_won = simulate_game(p_A if current_server_is_A else p_B)
        if current_server_is_A:
            games_A += server_won
            games_B += not server_won
        else:
            games_B += server_won
            games_A += not server_won
        if games_A >= 6 and games_A - games_B >= 2:
            return games_A + games_B, True, not current_server_is_A
        if games_B >= 6 and games_B - games_A >= 2:
            return games_A + games_B, False, not current_server_is_A
        current_server_is_A = not current_server_is_A


def simulate_match(p_A, p_B, with_sets=False):
    sets_A = sets_B = total = 0
    server_starts_A = (np.random.rand() < 0.5)          # fair coin for first server
    while sets_A < 3 and sets_B < 3:
        games, a_won, server_starts_A = simulate_set(p_A, p_B, server_starts_A)
        total += games
        sets_A += a_won
        sets_B += not a_won
    return (total, sets_A + sets_B) if with_sets else total


def p_game(p):
    """Exact probability of holding serve when each point is won w.p. p."""
    return (p**4 + 4*p**4*(1-p) + 10*p**4*(1-p)**2
            + 20*p**3*(1-p)**3 * (p**2 / (p**2 + (1-p)**2)))


N = 100_000
np.random.seed(42)
res = np.array([simulate_match(p_alc_serve, p_sin_serve, with_sets=True) for _ in range(N)])
sim_g, sim_s = res[:, 0], res[:, 1]

print("Mean games:", sim_g.mean())
print("Min games:", sim_g.min())
print("Max games:", sim_g.max())
print("Alcaraz hold prob:", p_game(p_alc_serve))
print("Sinner hold prob:", p_game(p_sin_serve))

p_A_mean = np.mean([w/t for w, t in zip(alc_won, alc_tot)])
p_S_mean = np.mean([w/t for w, t in zip(sin_won, sin_tot)])
print(f"Pooled: p_A = {p_alc_serve:.4f}, p_S = {p_sin_serve:.4f}")
print(f"Mean of ratios: p_A = {p_A_mean:.4f}, p_S = {p_S_mean:.4f}")

# the numbers the assignment script printed, asserted so a regression is loud
assert abs(sim_g.mean() - 41.72156) < 1e-9, sim_g.mean()
assert (sim_g.min(), sim_g.max()) == (21, 65)


# ------------------------------------------- exact pmf (validates the MC)
def tb_win_prob(pa, pb, a_first):
    """P(A wins a 7-point tie break), pa/pb = prob of winning a point on own serve."""
    @lru_cache(None)
    def f(a, b):
        if a >= 7 and a - b >= 2:
            return 1.0
        if b >= 7 and b - a >= 2:
            return 0.0
        if a >= 6 and b >= 6 and a == b:
            # from a tie past 6-6, points come in pairs (one on each player's serve)
            x, y = pa, 1 - pb
            return x*y / (x*y + (1-x)*(1-y))
        n = a + b
        blk = (n + 1) // 2
        a_srv = a_first if (n == 0 or blk % 2 == 0) else (not a_first)
        q = pa if a_srv else 1 - pb
        return q*f(a+1, b) + (1-q)*f(a, b+1)
    return f(0, 0)


def set_dist(pa, pb, a_first):
    """Distribution of (games in the set, A won it)."""
    hA, hB = p_game(pa), p_game(pb)
    out = {}

    def rec(a, b, prob):
        if prob == 0:
            return
        if (a >= 6 and a - b >= 2) or (b >= 6 and b - a >= 2):
            out[(a+b, a > b)] = out.get((a+b, a > b), 0) + prob
            return
        if a == 6 and b == 6:
            t = tb_win_prob(pa, pb, a_first)
            out[(13, True)] = out.get((13, True), 0) + prob*t
            out[(13, False)] = out.get((13, False), 0) + prob*(1-t)
            return
        a_srv = a_first if (a+b) % 2 == 0 else (not a_first)
        q = hA if a_srv else 1 - hB
        rec(a+1, b, prob*q)
        rec(a, b+1, prob*(1-q))

    rec(0, 0, 1.0)
    return out


def match_pmf(pa, pb):
    """Exact pmf of total games, and its split by number of sets."""
    sd = {True: set_dist(pa, pb, True), False: set_dist(pa, pb, False)}
    pmf = np.zeros(66)
    by_sets = np.zeros((66, 6))

    def rec(sa, sb, a_first, g, prob):
        if sa == 3 or sb == 3:
            pmf[g] += prob
            by_sets[g, sa+sb] += prob
            return
        for (gs, aw), pr in sd[a_first].items():
            rec(sa + aw, sb + (not aw), a_first if gs % 2 == 0 else (not a_first), g + gs, prob*pr)

    for af in (True, False):
        rec(0, 0, af, 0, 0.5)
    return pmf, by_sets


g = np.arange(66)
exact_pool, exact_pool_sets = match_pmf(p_alc_serve, p_sin_serve)
mc_pmf = np.bincount(sim_g, minlength=66)[:66] / N
tv = 0.5*np.abs(mc_pmf - exact_pool).sum()
print(f"exact pmf sums to {exact_pool.sum():.12f}, mean {(g*exact_pool).sum():.6f}, TV(MC, exact) = {tv:.6f}")
assert abs(exact_pool.sum() - 1) < 1e-9
assert tv < 0.02, tv          # sampling noise at N = 1e5 is ~0.008

# sensitivity variants
exact_mr, _ = match_pmf(p_A_mean, p_S_mean)
per_match = [match_pmf(aw/at, sw/st)[0]
             for aw, at, sw, st in zip(alc_won, alc_tot, sin_won, sin_tot)]
exact_mix = np.mean(per_match, axis=0)

np.random.seed(7)
pA_i = np.array(alc_won)/np.array(alc_tot)
pS_i = np.array(sin_won)/np.array(sin_tot)
idx = np.random.randint(0, 6, N)
mc_mix = np.bincount([simulate_match(pA_i[i], pS_i[i]) for i in idx], minlength=66)[:66] / N

stats = {}
for name, p in [("mc_pooled", mc_pmf), ("exact_pooled", exact_pool),
                ("exact_meanratio", exact_mr), ("exact_mixture", exact_mix), ("mc_mixture", mc_mix)]:
    m = (g*p).sum()
    stats[name] = dict(mean=float(m), sd=float(np.sqrt(((g-m)**2*p).sum())), mode=int(g[np.argmax(p)]))
emp = np.array(total_games)
stats["empirical"] = dict(mean=float(emp.mean()), sd=float(emp.std()), mode=None)

sets_frac = {str(k): float(np.mean(sim_s == k)) for k in (3, 4, 5)}
exact_sets = {str(k): float(exact_pool_sets[:, k].sum()) for k in (3, 4, 5)}
cdf = np.cumsum(mc_pmf)
pct = {str(int(x)): float(cdf[x-1] + 0.5*mc_pmf[x]) for x in emp}   # mid-percentile
print("sets (MC):", {k: round(v, 5) for k, v in sets_frac.items()})
print("sets (exact):", {k: round(v, 6) for k, v in exact_sets.items()})
print("mid-percentile of each observed match:", {k: round(v, 4) for k, v in pct.items()})
print("SE of a mean of 6 draws:", round(stats["mc_pooled"]["sd"]/np.sqrt(6), 4))

with path("summary.json").open("w", encoding="utf-8") as handle:
    json.dump(
        dict(
            stats=stats,
            sets_frac=sets_frac,
            exact_sets=exact_sets,
            pct=pct,
            tv=float(tv),
        ),
        handle,
        indent=2,
    )
np.savez(path("results.npz"), mc_pmf=mc_pmf, exact_pool=exact_pool, exact_pool_sets=exact_pool_sets,
         exact_mr=exact_mr, exact_mix=exact_mix, mc_mix=mc_mix, sim_s=sim_s, sim_g=sim_g)

# ----------------------------------------------------------------- figures
use_style()

# Figure 0 — exactly the figure the assignment script produced
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
vals, counts = np.unique(total_games, return_counts=True)
axes[0].stem(vals, counts/len(total_games), basefmt=" ")
axes[0].set_title("1(a) Empirical PMF of Total Games (N = 6 Matches)")
axes[0].set_xlabel("Total Games"); axes[0].set_ylabel("Probability")
axes[0].set_ylim(0, 0.25); axes[0].grid(True, alpha=0.3)
mv, mcnt = np.unique(sim_g, return_counts=True)
axes[1].bar(mv, mcnt/N, width=0.8, color="skyblue", edgecolor="black", alpha=0.7)
axes[1].set_title("1(b) Monte Carlo PMF of Total Games (100,000 Simulations)")
axes[1].set_xlabel("Total Games"); axes[1].set_ylabel("Probability")
axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(path("tennis_pmf_comparison.png"), dpi=200); plt.close()

# Figure 1 — 1(a)
fig, ax = plt.subplots(figsize=(7.2, 3.3))
ml, sl, _ = ax.stem(total_games, np.full(6, 1/6), basefmt=" ")
plt.setp(sl, color="#2c3e50", lw=2); plt.setp(ml, color=RED, ms=7)
for i, (x, k) in enumerate(zip(total_games, n_sets)):
    ax.annotate(f"M{i+1}\n{k} sets\n({x})", (x, 1/6), xytext=({0: -9, 4: 9}.get(i, 0), 6),
                textcoords="offset points", ha="center", fontsize=7.5)
ax.axvline(emp.mean(), ls="--", color="gray", lw=1)
ax.text(emp.mean()+0.4, 0.245, f"mean = {emp.mean():.1f}", fontsize=8, color="gray")
ax.set_xlim(17, 66); ax.set_ylim(0, 0.26); ax.set_xticks(range(18, 67, 4))
ax.set_xlabel("Total games in match"); ax.set_ylabel(r"$\hat p(k)$")
ax.set_title("1(a)  Empirical pmf from the 6 observed matches (each value has mass 1/6)")
ax.grid(alpha=0.25); plt.tight_layout(); plt.savefig(path("fig1a.png"), dpi=220); plt.close()

# Figure 2 — 1(b), stacked by number of sets, with the exact pmf overlaid
fig, ax = plt.subplots(figsize=(7.2, 3.8))
parts = {k: np.bincount(sim_g[sim_s == k], minlength=66)[:66]/N for k in (3, 4, 5)}
ax.bar(g, parts[3], color=C3, width=0.85, label=f"3 sets ({sets_frac['3']:.1%})")
ax.bar(g, parts[4], bottom=parts[3], color=C4, width=0.85, label=f"4 sets ({sets_frac['4']:.1%})")
ax.bar(g, parts[5], bottom=parts[3]+parts[4], color=C5, width=0.85, label=f"5 sets ({sets_frac['5']:.1%})")
nz = exact_pool > 1e-6
ax.plot(g[nz], exact_pool[nz], "o", ms=3, mfc="white", mec="k", mew=0.8, label="exact pmf (Markov-chain DP)")
ax.plot(total_games, np.zeros(6)-0.002, "^", color=RED, ms=8, clip_on=False, label="observed matches")
ax.axvline(sim_g.mean(), ls="--", color="gray", lw=1)
ax.text(sim_g.mean()+0.4, 0.052, f"mean = {sim_g.mean():.2f}\nsd = {sim_g.std():.2f}", fontsize=8, color="dimgray")
ax.set_xlim(17, 66); ax.set_ylim(-0.004, 0.058); ax.set_xticks(range(18, 67, 4))
ax.set_xlabel("Total games in match"); ax.set_ylabel("Probability")
ax.set_title(r"1(b)  Monte Carlo pmf, $N=10^5$, $p_A=0.6251$, $p_S=0.6191$ (pooled)")
ax.legend(fontsize=8, frameon=False, loc="upper left"); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(path("fig1b.png"), dpi=220); plt.close()

# Figure 3 — CDF comparison and sensitivity
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2))
ax = axs[0]
ax.step(g, np.cumsum(mc_pmf), where="post", color=C4, lw=1.8, label="Monte Carlo (pooled)")
xs = np.sort(emp)
ax.step(np.r_[17, xs, 66], np.r_[0, np.arange(1, 7)/6, 1], where="post", color=RED, lw=1.5,
        label="empirical (n = 6)")
ax.set_xlim(17, 66); ax.set_xlabel("Total games"); ax.set_ylabel("CDF")
ax.set_title("CDF comparison"); ax.legend(fontsize=7.5, frameon=False); ax.grid(alpha=0.25)
ax = axs[1]
ax.plot(g[nz], exact_pool[nz], "-", color=C4, lw=1.6, label="pooled $p$ (exact)")
ax.plot(g[exact_mr > 1e-6], exact_mr[exact_mr > 1e-6], "--", color="#e67e22", lw=1.3,
        label="mean of ratios (exact)")
ax.plot(g[exact_mix > 1e-6], exact_mix[exact_mix > 1e-6], "-", color="#27ae60", lw=1.4,
        label="per-match mixture (exact)")
ax.plot(g[mc_mix > 0], mc_mix[mc_mix > 0], ".", color="#27ae60", ms=3, label="per-match mixture (MC)")
ax.set_xlim(17, 66); ax.set_xlabel("Total games"); ax.set_ylabel("Probability")
ax.set_title("Sensitivity to how $p$ is estimated")
ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(path("fig1c.png"), dpi=220); plt.close()
print("p1_tennis.py done")
