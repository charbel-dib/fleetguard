"""Headless figures and a compact report generated from measured comparison outputs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay

from fleetguard.diagnostics import reliability_table


def _save(figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def write_comparison_report(run_dir, summary, prediction, metadata, errors) -> None:
    figures = run_dir / "figures"
    figures.mkdir()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(summary["model"][::-1], summary["cost_per_row"][::-1], color="#2563eb")
    ax.set_xlabel("Out-of-fold challenge cost per observation")
    ax.set_title("Train-only comparison — thresholds tuned on separate observations")
    ax.grid(axis="x", alpha=0.2)
    _save(fig, figures / "cv_cost.png")
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(
        prediction["target"], prediction["score"], name=metadata["model"], ax=ax
    )
    ax.set_title("Retained development validation — frozen champion")
    _save(fig, figures / "precision_recall.png")
    table = reliability_table(prediction["target"], prediction["score"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Reference")
    ax.plot(table["mean_score"], table["positive_rate"], "o-", label=metadata["model"])
    ax.set(
        xlabel="Mean positive-class score",
        ylabel="Observed positive fraction",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend()
    ax.set_title("Equal-width reliability bins; sparse bins can be unstable")
    _save(fig, figures / "reliability.png")
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        prediction["target"],
        prediction["prediction"],
        labels=[0, 1],
        display_labels=["Non-APS", "APS"],
        colorbar=False,
        ax=ax,
    )
    ax.set_title("Frozen threshold on retained validation")
    _save(fig, figures / "confusion.png")
    metrics = metadata["validation"]
    lines = [
        "# FleetGuard — train-only comparison",
        "",
        f"Run: `{run_dir.name}`.",
        "",
        "Candidates are compared using scoring folds inside the development partition.",
        "Fitting, calibration and threshold-tuning roles are disjoint within every fold.",
        "",
        "| Candidate | OOF cost | Cost/row | Mean AP | Mean recall | Fold cost/row SD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.cost} | {row.cost_per_row:.4f} | "
            f"{row.mean_average_precision:.4f} | "
            f"{row.mean_recall:.4f} | {row.std_fold_cost_per_row:.4f} |"
        )
    lines.extend(
        [
            "",
            "![CV cost](figures/cv_cost.png)",
            "",
            "## Frozen champion",
            "",
            f"Selected by development CV: `{metadata['model']}`.",
            f"Fitting/reserved-calibration/threshold rows: {metadata['training_rows']}/"
            f"{metadata['calibration_rows']}/{metadata['threshold_rows']}.",
            f"Retained validation rows: {metadata['validation_rows']}; "
            f"threshold: {metadata['threshold']:.8g}.",
            "",
            f"Cost: **{metrics['cost']}**; FP: **{metrics['fp']}**; FN: **{metrics['fn']}**; "
            f"recall: **{metrics['recall']:.4f}**; precision: **{metrics['precision']:.4f}**.",
            "",
            "![Precision recall](figures/precision_recall.png)",
            "",
            "![Confusion](figures/confusion.png)",
            "",
            "![Reliability](figures/reliability.png)",
            "",
            "## Errors by missingness",
            "",
            "| Missing fraction | Observations | Positives | FP | FN | Cost/row |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in errors.itertuples(index=False):
        lines.append(
            f"| {row.missing_lower:.0%}–{row.missing_upper:.0%} | {row.rows} | {row.positive} | "
            f"{row.fp} | {row.fn} | {row.cost_per_row:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "Fold standard deviations describe variation across correlated CV folds.",
            "They are not confidence intervals.",
            "Validation was inspected during increment 1; it is not a new blind test.",
            "Only the CV-selected champion is scored there. No official test is evaluated.",
            "Increment-1 models had different fitting sample sizes.",
            "Direct comparison of their scores with this protocol is inappropriate.",
            "Sigmoid calibration uses a separate role; its quality must still be measured.",
            "",
        ]
    )
    (run_dir / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")
