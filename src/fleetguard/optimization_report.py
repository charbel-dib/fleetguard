"""Search and data-size learning curves from executed experiments only."""

import matplotlib.pyplot as plt
import pandas as pd

from fleetguard.comparison_report import _save, write_comparison_report


def write_optimization_report(run_dir, summary, prediction, metadata, errors):
    write_comparison_report(run_dir, summary, prediction, metadata, errors)
    figures = run_dir / "figures"
    trials = pd.read_csv(run_dir / "trials.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    for family, rows in trials.groupby("family"):
        rows = rows.sort_values("trial")
        ax.plot(rows["trial"], rows["cost_per_row"].cummin(), "o-", label=family)
    ax.set(
        xlabel="Sequential Optuna trial",
        ylabel="Best search CV cost per observation",
        title="Fixed search budget — selection scores, not an independent test",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    _save(fig, figures / "search_budget.png")
    learning = pd.read_csv(run_dir / "learning_curves.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, rows in learning.groupby("model"):
        grouped = rows.groupby("fraction")
        sizes = grouped["training_rows"].mean()
        costs = grouped["cost_per_row"].mean()
        ax.plot(sizes, costs, "o-", label=name)
    ax.set(
        xlabel="Mean fitted rows per scoring fold",
        ylabel="Mean scoring cost per observation",
        title="Data-size learning curves — fixed selected parameters",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    _save(fig, figures / "learning_curves.png")
    epochs = pd.read_csv(run_dir / "candidates/mlp_tuned/epoch_history.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    for fold, rows in epochs.query("stage == 'epoch_selection'").groupby("fold"):
        ax.plot(rows["epoch"], rows["early_stopping_loss"], label=f"Fold {fold + 1}")
    ax.set(
        xlabel="Epoch",
        ylabel="Weighted BCE on fit-only holdout",
        title="MLP epoch selection — calibration and scoring labels excluded",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    _save(fig, figures / "mlp_epochs.png")
    stability = pd.read_csv(run_dir / "stability.csv")
    original = (run_dir / "comparison_report.md").read_text(encoding="utf-8")
    original = original.replace(
        "# FleetGuard — train-only comparison", "# FleetGuard — budgeted optimization"
    )
    additions = [
        "## Search protocol and limitations",
        "",
        "Optuna trials reuse the development scoring folds. These CV scores are selection scores",
        "and can be optimistic after search. These are not nested-CV or final-test estimates.",
        "The candidate is frozen before diagnostics and retained-validation scoring.",
        "Stability changes group split seeds with model seed fixed; it does not retune parameters.",
        "Data-size curves vary fit rows only. Calibration, threshold and scoring roles stay fixed.",
        "MLP epochs use a group-aware holdout inside fit; then all fit rows are refitted.",
        "Sigmoid calibration subsequently uses the separate calibration role.",
        "Oversampling is fitted inside the fit-only pipeline. Other roles keep class frequencies.",
        "",
        "![Search budget](figures/search_budget.png)",
        "",
        "![Data-size curves](figures/learning_curves.png)",
        "",
        "![Epoch selection](figures/mlp_epochs.png)",
        "",
        "## Partition-seed stability",
        "",
        "| Candidate | CV seed | Cost/row | Mean AP | FP | FN |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in stability.itertuples(index=False):
        additions.append(
            f"| {row.model} | {row.seed} | {row.cost_per_row:.4f} | "
            f"{row.mean_average_precision:.4f} | {row.fp} | {row.fn} |"
        )
    controls = pd.read_csv(run_dir / "control_status.csv").fillna("")
    additions.extend(
        [
            "",
            "## Control status",
            "",
            "Only controls converged on every scoring fold enter the leaderboard.",
            "",
            "| Control | Status | Iteration budget |",
            "|---|---|---:|",
        ]
    )
    for row in controls.itertuples(index=False):
        additions.append(f"| {row.model} | {row.status} | {row.max_iter} |")
    (run_dir / "optimization_report.md").write_text(
        original + "\n" + "\n".join(additions) + "\n", encoding="utf-8"
    )
