"""Shared paths and plotting style for the probability assignment."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
REPORT = ROOT / "report"
SOLUTIONS = ROOT / "solutions"

for directory in (OUT, REPORT, SOLUTIONS):
    directory.mkdir(parents=True, exist_ok=True)


def path(name: str) -> Path:
    """Return a path inside the generated-output directory."""

    return OUT / name


def use_style() -> None:
    """Apply a compact NYU-inspired style to all figures."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titlecolor": "#2B1739",
            "axes.labelcolor": "#2B1739",
            "axes.edgecolor": "#6B6570",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


# NYU purple is kept under the historical name RED so the analysis scripts do
# not need cosmetic-only changes to every plotting call.
RED, BLUE, GRAY, GREEN = "#57068C", "#2471A3", "#7F8C8D", "#2E7D32"
C3, C4, C5 = "#D8C3E7", "#8B50B6", "#3C1053"
