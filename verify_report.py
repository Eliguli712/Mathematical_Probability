"""Cross-check the PDF's numerical claims against the generated JSON files."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
PDF = ROOT / "report" / "Mathematical_Probability_Homework.pdf"


def load_json(name: str) -> dict:
    with (OUT / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def pdf_scientific(value: float, precision: int) -> str:
    """Format scientific notation as pdftotext extracts it from LaTeX."""

    mantissa, raw_exponent = f"{value:.{precision}e}".split("e")
    exponent = str(int(raw_exponent)).replace("-", "−")
    return f"{mantissa} × 10{exponent}"


def main() -> int:
    if shutil.which("pdftotext") is None:
        print("pdftotext is required for report verification.", file=sys.stderr)
        return 2
    if not PDF.exists():
        print(f"Missing report: {PDF.relative_to(ROOT)}. Run python3 run_all.py first.", file=sys.stderr)
        return 2

    completed = subprocess.run(
        ["pdftotext", str(PDF), "-"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode:
        print(completed.stderr, file=sys.stderr)
        return completed.returncode

    text = " ".join(completed.stdout.split())
    summary = load_json("summary.json")
    fire = load_json("fire.json")
    kl = load_json("kl.json")
    kl2 = load_json("kl2.json")
    checks: list[tuple[str, str]] = []

    def claim(label: str, value: str) -> None:
        checks.append((label, value))

    # Problem 1
    claim("observed totals", "39, 56, 45, 59, 40, 34")
    claim("pooled serve probabilities", "0.6251 (532/851)")
    claim("MC mean", f"{summary['stats']['mc_pooled']['mean']:.2f}")
    claim("MC sd", f"{summary['stats']['mc_pooled']['sd']:.2f}")
    claim("simulated range", "21–65")
    claim("exact mean", f"{summary['stats']['exact_pooled']['mean']:.3f}")
    claim("total variation distance", f"{summary['tv']:.4f}")
    claim(
        "set probabilities",
        (
            f"{summary['exact_sets']['3']:.3f}, "
            f"{summary['exact_sets']['4']:.3f}, "
            f"{summary['exact_sets']['5']:.3f}"
        ),
    )
    claim("MC set fractions", f"three sets ({summary['sets_frac']['3']:.1%})")
    claim("mean-of-ratios variant", f"{summary['stats']['exact_meanratio']['mean']:.2f}")
    claim("per-match mixture variant", f"{summary['stats']['exact_mixture']['mean']:.1f}")
    claim("mixture mode", f"mode {summary['stats']['exact_mixture']['mode']}")
    claim("empirical mean", f"{summary['stats']['empirical']['mean']}")
    claim("observation percentiles", f"{min(summary['pct'].values()):.0%}")

    # Problem 3
    claim("alpha_hat", f"{fire['alpha_hat']:.6f}")
    claim("lambda_hat", f"{fire['lambda_hat']:.6f}")
    claim("empirical tail ratio 10->100", f"{fire['R_emp'][0]:.1f}")
    claim("empirical tail ratio 100->1000", f"{fire['R_emp'][1]:.1f}")
    claim("Pareto tail ratio", f"{fire['R_par']:.1f}")
    claim(
        "Pareto catastrophic probability",
        pdf_scientific(fire["P_par"], 2),
    )
    claim("Pareto predicted count", f"{fire['P_par'] * fire['Ntot']:.2f}")
    claim("true predicted count", f"{fire['P_true'] * fire['Ntot']:.2f}")

    # Problem 4
    claim("Gaussian KL closed form", f"{kl['kc']:.4f}")
    claim("Gaussian KL Monte Carlo", f"{kl['mc']:.4f}")
    claim("Gaussian KL reverse", f"{kl['rev']:.4f}")
    claim("MLE grid maximizer", f"{kl2['th_ml']:.4f}")
    claim("closed-form binomial MLE", f"{kl2['closed']:.4f}")
    claim("identity residual", pdf_scientific(kl2["gap"], 1))
    claim("empirical entropy", f"{kl2['H']:.4f}")
    claim("4(b) relative error", pdf_scientific(kl2["rel1e3"], 2))
    claim("4(b) fitted slope", f"{kl2['slope']:.2f}")

    width = max(len(label) for label, _ in checks)
    failed = 0
    for label, value in checks:
        ok = value in text
        failed += not ok
        print(f"{'OK  ' if ok else 'FAIL'}  {label.ljust(width)}  {value}")
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
