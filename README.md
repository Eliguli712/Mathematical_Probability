# Mathematical Probability

Reproducible solutions for a four-part probability assignment covering tennis
match lengths, Pareto tails, catastrophic-fire extrapolation, and
Kullback–Leibler divergence.

The repository contains the complete Python analysis, a readable Markdown
solution, the formal NYU-style LaTeX source, generated figures and numerical
summaries, and the final homework PDF.

## Submission files

| Deliverable | File |
|---|---|
| Formal homework PDF | [Mathematical_Probability_Homework.pdf](report/Mathematical_Probability_Homework.pdf) |
| LaTeX source | [Mathematical_Probability_Homework.tex](report/Mathematical_Probability_Homework.tex) |
| Markdown solution | [tennis_pareto_kl_solutions.md](solutions/tennis_pareto_kl_solutions.md) |
| Reproducible Python entry point | [run_all.py](run_all.py) |
| Numerical report verifier | [verify_report.py](verify_report.py) |
| Latest deterministic run log | [run_log.txt](out/run_log.txt) |

## Main results

| Topic | Result |
|---|---|
| Pooled serve-point estimates | Alcaraz 532/851 = 0.6251; Sinner 564/911 = 0.6191 |
| Simulated match length | mean 41.72 games; standard deviation 8.74 |
| Exact match-length calculation | mean 41.735 games |
| Monte Carlo validation | total-variation distance 0.0080 |
| Exact match duration | 3 sets 0.251; 4 sets 0.375; 5 sets 0.374 |
| Pareto fire-size fit | m̂ = 10.000078; α̂ = 1.515540 |
| Catastrophic-fire prediction | 1.40 Pareto-predicted fires versus 1.61 under the generating model |
| Gaussian KL check | closed form 1.7566; Monte Carlo 1.7540 ± 0.0058 |
| MLE–KL identity | residual 6.7 × 10⁻¹⁶ |

## Problems covered

1. **Tennis probability mass functions**
   - Constructs the empirical pmf from six observed matches.
   - Simulates 100,000 best-of-five matches point by point.
   - Computes the same pmf exactly with dynamic programming.
   - Checks sensitivity to alternative estimators of serve-point probability.

2. **Pareto distribution**
   - Derives the density and survival function.
   - Proves scale invariance of the tail.
   - Derives the maximum-likelihood estimators of the scale and shape.
   - Compares Pareto and exponential tails graphically.

3. **Catastrophic fires**
   - Explains why a one-year empirical tail cannot extrapolate.
   - Diagnoses exponential versus power-law behavior using tail ratios.
   - Fits a Pareto model to a deterministic synthetic sample.
   - Predicts the expected number of fires exceeding 100,000 acres.

4. **Kullback–Leibler divergence**
   - Proves nonnegativity using Jensen's inequality.
   - Derives the multivariate-Gaussian closed form.
   - Shows that maximum likelihood minimizes empirical-to-model KL.
   - Derives the local weighted least-squares approximation.

## Quick start

Python 3.10 or later is recommended.

    python3 -m venv .venv
    source .venv/bin/activate
    python3 -m pip install -r requirements.txt
    python3 run_all.py
    python3 verify_report.py

The report stage requires a TeX installation with either latexmk or pdflatex.
On Debian or Ubuntu, a suitable setup is:

    sudo apt-get install latexmk texlive-latex-extra texlive-fonts-recommended

To run only the numerical analysis, execute the first four scripts directly;
LaTeX is needed only by src/build_report.py.

## Reproducibility

Every random component has a fixed seed:

| Script | Seed | Purpose |
|---|---:|---|
| src/p1_tennis.py | 42 | 100,000 pooled-probability match simulations |
| src/p1_tennis.py | 7 | per-match-mixture sensitivity simulation |
| src/p3_fires.py | 42 | 5,000 synthetic fire sizes |
| src/p4_kl.py | 0 | four-dimensional Gaussian KL check |
| src/p4_kl.py | 11 | binomial MLE/KL demonstration |

The tennis script contains regression assertions for the reference simulation
and an independent exact-pmf calculation. The KL script checks its closed form
against Monte Carlo and verifies the likelihood/KL identity to machine
precision.

After a complete run, verify_report.py extracts text from the final PDF with
pdftotext and checks 31 reported numerical claims against the JSON results.

## Model consistency correction

The fire analysis uses a single unshifted exponential model throughout:

$$
f(a)=\widehat\lambda e^{-\widehat\lambda a},
\qquad
\Pr(A>a)=e^{-\widehat\lambda a},
\qquad
\widehat\lambda=\frac{1}{\overline a}.
$$

This removes a mismatch in the earlier script, which estimated an unshifted
exponential rate but plotted a shifted density. The correction does not change
the conclusion: the exponential tail still underflows at 100,000 acres, while
the fitted Pareto model gives a small but nonzero prediction.

## Repository structure

    .
    ├── README.md
    ├── requirements.txt
    ├── run_all.py
    ├── verify_report.py
    ├── src/
    │   ├── common.py
    │   ├── p1_tennis.py
    │   ├── p2_pareto.py
    │   ├── p3_fires.py
    │   ├── p4_kl.py
    │   └── build_report.py
    ├── solutions/
    │   └── tennis_pareto_kl_solutions.md
    ├── report/
    │   ├── Mathematical_Probability_Homework.tex
    │   └── Mathematical_Probability_Homework.pdf
    └── out/
        ├── fig*.png
        ├── *.json
        └── run_log.txt

Large simulation arrays are generated locally but are intentionally omitted
from version control. The compact figures and JSON summaries required to audit
the submitted report are included.

## Notes on interpretation

- The tennis model treats points as independent and keeps serve-point
  probabilities fixed across matches. It does not model surface, fatigue,
  momentum, or match-specific form.
- The fire data are synthetic and are used to demonstrate tail-model behavior,
  not to make a real-world forecast.
- KL divergence is measured in nats because natural logarithms are used.

Author: Eliguli Han · New York University
