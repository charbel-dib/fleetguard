"""Descriptive plots and model card; these never select a model or threshold."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fleetguard.io import read_json, write_json


def write_release_report(directory):
    decision = read_json(directory / "decision.json")
    metrics = read_json(directory / "validation_metrics.json")
    calibration = pd.read_csv(directory / "calibration_cv.csv")
    policies = pd.read_csv(directory / "validation_policies.csv")
    importance = pd.read_csv(directory / "permutation_columns.csv")
    benchmark = read_json(directory / "benchmark.json")
    final_path = directory / "final_test/metrics.json"
    final = read_json(final_path) if final_path.exists() else None
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for label in ("raw", "release"):
        table = pd.read_csv(directory / f"reliability_{label}.csv")
        axes[0].plot(table.mean_score, table.positive_rate, "o-", label=label)
    axes[0].plot([0, 1], [0, 1], "--", color="gray", alpha=0.6)
    axes[0].set(
        xlabel="Mean predicted score",
        ylabel="Observed APS fraction",
        title="Validation reliability (10 bins)",
    )
    axes[0].legend()
    axes[1].bar(calibration.method, calibration.brier_score, color=["#607d8b", "#167d8d"])
    axes[1].set(ylabel="Brier score (lower is better)", title="Calibration-role selection CV")
    fig.savefig(figures / "calibration.png", dpi=160)
    plt.close(fig)

    top = importance.head(15).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), constrained_layout=True)
    axes[0].barh(
        top.feature,
        top.cost_increase_per_row_mean,
        xerr=top.cost_increase_per_row_std,
        color="#167d8d",
    )
    axes[0].set(xlabel="Increase in cost per row", title="Fixed-threshold permutation audit")
    axes[1].barh(
        top.feature,
        top.average_precision_drop_mean,
        xerr=top.average_precision_drop_std,
        color="#d27d32",
    )
    axes[1].set(xlabel="Decrease in average precision", title="Same 15 features; ranking impact")
    fig.savefig(figures / "permutation.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for row in policies.itertuples(index=False):
        ax.scatter(row.inspection_rate * 100, row.recall * 100, s=70)
        ax.annotate(
            f"{row.policy}\ncost {row.cost:,}",
            (row.inspection_rate * 100, row.recall * 100),
            xytext=(6, -22 if row.policy == "challenge_cost" else 8),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set(
        xlabel="Validation rows flagged (%)",
        ylabel="Validation APS recall (%)",
        title="Policies frozen on the threshold role",
    )
    ax.margins(x=0.25, y=0.25)
    ax.grid(alpha=0.2)
    fig.savefig(figures / "inspection.png", dpi=160)
    plt.close(fig)

    result = metrics["release"] if final is None else final["metrics"]
    matrix = np.array([[result["tn"], result["fp"]], [result["fn"], result["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    ax.imshow(
        matrix, cmap="Blues", norm=matplotlib.colors.LogNorm(vmin=1, vmax=max(matrix.max(), 2))
    )
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:,}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > matrix.max() / 3 else "black",
                fontsize=15,
            )
    ax.set(
        xticks=[0, 1],
        xticklabels=["neg", "pos"],
        yticks=[0, 1],
        yticklabels=["neg", "pos"],
        xlabel="Prediction",
        ylabel="Observed label",
        title="Official test" if final else "Retained validation",
    )
    fig.savefig(figures / "confusion.png", dpi=160)
    plt.close(fig)

    batches = pd.DataFrame(benchmark["batches"])
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    positions = np.arange(len(batches))
    ax.bar(positions - 0.18, batches.median_ms, width=0.36, label="median", color="#167d8d")
    ax.bar(positions + 0.18, batches.p95_ms, width=0.36, label="p95", color="#d27d32")
    ax.set(
        xticks=positions,
        xticklabels=batches.batch_rows,
        xlabel="Batch rows",
        ylabel="Milliseconds",
        title="Local CPU inference; HTTP excluded",
    )
    ax.legend()
    fig.savefig(figures / "latency.png", dpi=160)
    plt.close(fig)

    rows = [
        "# FleetGuard — frozen model card",
        "",
        f"Release: `{directory.name}`. Frozen at `{decision['frozen_at_utc']}`.",
        f"Base: `{decision['source_champion']}` from `{decision['source_run']}`; no base refit.",
        f"Calibration: **{decision['calibration']}**. Threshold: **{decision['threshold']:.12g}**.",
        "",
        "## Task and intended use",
        "",
        "Portfolio/research demonstration of cost-aware APS diagnostic "
        "classification on the public Scania snapshot.",
        "pos means APS-associated failure; neg means another component failure, not "
        "a healthy truck.",
        "The 170 measurements are anonymized; timestamps, vehicle identity and a "
        "forecasting horizon are absent.",
        "This model is not validated for maintenance deployment, safety decisions or "
        "new fleet populations.",
        "",
        "## Selection and separation",
        "",
        "The base champion/parameters were selected by reused development scoring CV in update 03.",
        "That search has selection optimism and is not nested CV. This release "
        "retains that base pipeline.",
        "Raw versus sigmoid is selected by pooled Brier on grouped OOF predictions "
        "inside the calibration role.",
        "The selected map is fit on that complete role. Both roles remain disjoint "
        "from the base fit and threshold role.",
        "Calibration rows previously contributed to development hyperparameter "
        "search, so this is not independent model assessment.",
        "A lower Brier combines reliability and discrimination; it does not prove "
        "universal probability calibration.",
        "The threshold minimizes 10FP + 500FN on the threshold role; ties favor "
        "fewer FN, FP, then higher threshold.",
        "The selected policy is challenge_cost. Validation diagnostics cannot change "
        "this decision.",
        "Validation was observed in earlier increments; only the official test "
        "remains the final untouched benchmark.",
        "",
        "| Calibration-role OOF method | Rows | Brier | Average precision |",
        "|---|---:|---:|---:|",
    ]
    for row in calibration.itertuples(index=False):
        rows.append(
            f"| {row.method} | {row.rows} | {row.brier_score:.6f} | {row.average_precision:.6f} |"
        )
    rows += [
        "",
        "## Measured outcomes",
        "",
        "| Partition / pipeline | FP | FN | Recall | Precision | AP | Brier | Cost | Flagged |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    observed = [
        ("Validation / source raw", metrics["source_raw"]),
        ("Validation / release", metrics["release"]),
    ]
    if final:
        observed.append(("Official test / frozen release", final["metrics"]))
    for label, value in observed:
        rows.append(
            f"| {label} | {value['fp']} | {value['fn']} | {value['recall']:.2%} | "
            f"{value['precision']:.2%} | {value['average_precision']:.6f} | "
            f"{value['brier_score']:.6f} | {value['cost']} | {value['inspection_rate']:.2%} |"
        )
    rows += [
        "",
        "Costs use challenge units, not euros. Validation and official test have "
        "different prevalence.",
    ]
    if final:
        rows += [
            f"Official test rows: {final['metrics']['rows']}; "
            f"positives: {final['metrics']['tp'] + final['metrics']['fn']}.",
            "Exact feature rows also found in official train: "
            f"{final['feature_rows_also_present_in_official_train']}.",
            "The official test was scored once after freeze; no model or threshold was "
            "revised from its outcomes.",
        ]
        interval = read_json(directory / "final_test/uncertainty.json")
        rows += [
            "",
            "95% conditional intervals (500 bootstrap resamples of complete "
            "duplicate-feature groups):",
        ]
        for name, limits in interval["intervals"].items():
            rows.append(f"- {name}: [{limits['lower']:.6f}, {limits['upper']:.6f}]")
        rows += [
            "These intervals exclude model training/selection uncertainty and do not "
            "prove transport to other fleets."
        ]
    else:
        rows += [
            "Official test evaluation pending; no official test predictions were "
            "computed by the audit."
        ]
    rows += [
        "",
        "## Inspection scenarios on retained validation",
        "",
        "| Policy | Budget on threshold role | Actual validation flagged | Recall | Cost |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in policies.itertuples(index=False):
        budget = (
            "none"
            if pd.isna(row.budget_on_threshold_role)
            else f"{row.budget_on_threshold_role:.0%}"
        )
        rows.append(
            f"| {row.policy} | {budget} | {row.inspection_rate:.2%} | "
            f"{row.recall:.2%} | {row.cost} |"
        )
    rows += [
        "",
        "These are frozen-threshold scenarios, not hard capacity guarantees on new batches.",
        "The test uses only challenge_cost. The budget policies are not selected or "
        "compared using official-test labels.",
        "",
        "## Error analysis and interpretation",
        "",
        "Local validation_errors.csv records FP/FN row positions, scores, threshold "
        "margin and missingness.",
        "error_sensor_summary.csv reports sensor medians/nonmissing counts by "
        "outcome, without inventing causal explanations.",
        "Permutation importance uses every retained validation row and a fixed "
        "decision threshold; repeats measure shuffle variability.",
        "It does not measure training uncertainty. Correlated sensors can mask or "
        "share importance; shuffling can break plausible combinations.",
        "Multi-column prefixes are also shuffled jointly; these are naming groups, "
        "not identified physical subsystems.",
        "No feature is removed, no model is refit and no threshold is selected from "
        "these diagnostics.",
        "",
        "## CPU resource benchmark",
        "",
        f"Platform: {benchmark['platform']}; Python {benchmark['python']}; "
        f"logical CPUs {benchmark['logical_cpus']}.",
        f"One thread; {benchmark['batches'][0]['repeats']} measured repeats "
        "after one warm-up per batch.",
        f"Pipeline: {benchmark['pipeline_bytes']:,} bytes; "
        f"load: {benchmark['load_seconds']:.4f} s.",
        f"Whole-process peak RSS: {benchmark['process_peak_rss_bytes'] / 2**20:.2f} MiB; "
        f"traced Python peak: {benchmark['python_traced_peak_bytes'] / 2**20:.2f} MiB.",
        "RSS includes interpreter, imports and native allocations; it is not the "
        "model's incremental memory.",
        "Latency covers DataFrame-to-score only, excludes CSV/HTTP/network, and is "
        "specific to this environment.",
        "",
        "| Batch | Median ms | p95 ms | Rows/s at median |",
        "|---:|---:|---:|---:|",
    ]
    for row in batches.itertuples(index=False):
        rows.append(
            f"| {row.batch_rows} | {row.median_ms:.3f} | {row.p95_ms:.3f} | "
            f"{row.rows_per_second_at_median:.0f} |"
        )
    rows += [
        "",
        "## Artifact and limitations",
        "",
        "The persisted pipeline includes preprocessing, optional calibration, "
        "ordered schema and runtime versions.",
        "freeze.json hashes the pipeline, metadata, decision, policies, "
        "configuration, source and role manifests.",
        "Runtime verifies that freeze before inference/final evaluation. These "
        "hashes detect changes; they are not signatures.",
        "Only load trusted locally created joblib artifacts. Data/model files remain "
        "local and are not committed.",
        "The final-evaluation receipt is shared across runs using the same raw-data "
        "directory and prevents accidental repeat scoring.",
        "It is a local guard, not an access-control system; deleting receipts or "
        "copying raw data can bypass it.",
        "Release monitoring, drift thresholds, production SLAs and external "
        "validation remain future work.",
        "",
    ]
    (directory / "model_card.md").write_text("\n".join(rows), encoding="utf-8")
    write_json(
        directory / "summary.json",
        {
            "release_id": directory.name,
            "decision": decision,
            "validation": metrics,
            "calibration_cv": calibration.to_dict(orient="records"),
            "benchmark": benchmark,
            "official_test": final,
        },
    )
