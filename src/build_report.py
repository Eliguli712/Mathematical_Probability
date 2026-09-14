"""Compile and atomically publish the NYU-style LaTeX homework report."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from common import OUT, REPORT, ROOT


TEX = REPORT / "Mathematical_Probability_Homework.tex"
PDF = REPORT / "Mathematical_Probability_Homework.pdf"
REQUIRED_OUTPUTS = (
    "tennis_pmf_comparison.png",
    "fig1a.png",
    "fig1b.png",
    "fig1c.png",
    "fig2.png",
    "fig_fire_orig.png",
    "fig3.png",
    "fig4.png",
    "fig5.png",
    "fig6.png",
    "fig7.png",
    "summary.json",
    "fire.json",
    "kl.json",
    "kl2.json",
)


def ensure_inputs() -> None:
    """Fail early with a useful message when an upstream stage is missing."""

    missing = [name for name in REQUIRED_OUTPUTS if not (OUT / name).exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Missing generated outputs: {joined}")
    if not TEX.exists():
        raise FileNotFoundError(f"Missing LaTeX source: {TEX}")


def run_latex(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        cwd=REPORT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)


def is_complete_pdf(candidate: Path) -> bool:
    """Check the header and trailer before publishing a compiled PDF."""

    if not candidate.is_file() or candidate.stat().st_size < 10_000:
        return False
    content = candidate.read_bytes()
    return (
        content.startswith(b"%PDF-")
        and b"startxref" in content[-4096:]
        and b"%%EOF" in content[-1024:]
    )


def compile_report() -> None:
    """Compile in a temporary directory, then publish with an atomic rename."""

    with tempfile.TemporaryDirectory(prefix="math-probability-pdf-") as temp_name:
        temp_dir = Path(temp_name)
        temp_pdf = temp_dir / PDF.name

        latexmk = shutil.which("latexmk")
        if latexmk:
            run_latex(
                [
                    latexmk,
                    "-pdf",
                    "-silent",
                    f"-outdir={temp_dir}",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-file-line-error",
                    TEX.name,
                ]
            )
        else:
            pdflatex = shutil.which("pdflatex")
            if pdflatex is None:
                raise RuntimeError(
                    "A TeX distribution is required (latexmk or pdflatex was not found)."
                )
            command = [
                pdflatex,
                f"-output-directory={temp_dir}",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-file-line-error",
                TEX.name,
            ]
            run_latex(command)
            run_latex(command)

        if not is_complete_pdf(temp_pdf):
            raise RuntimeError("The TeX compiler produced an incomplete PDF.")

        staging = REPORT / f".{PDF.name}.tmp"
        shutil.copy2(temp_pdf, staging)
        staging.replace(PDF)


def main() -> int:
    try:
        ensure_inputs()
        compile_report()
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"report build failed: {exc}", file=sys.stderr)
        return 1

    if not is_complete_pdf(PDF):
        print("report build failed: the published PDF is incomplete", file=sys.stderr)
        return 1

    print(f"built {PDF.relative_to(ROOT)} ({PDF.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
