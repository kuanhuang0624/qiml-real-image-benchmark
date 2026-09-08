"""Generate the twelve required figures from saved CSV results only."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qiml_benchmark.paths import workspace_root
ROOT = workspace_root()
FORMATS = {"png": {"dpi": 300, "metadata": {"Software": "real_image_qml_benchmark_2026"}},
           "pdf": {"metadata": {"Creator": "real_image_qml_benchmark_2026", "CreationDate": None, "ModDate": None}},
           "svg": {"metadata": {"Date": None, "Creator": "real_image_qml_benchmark_2026"}}}
DATASETS = ["mnist", "fashion_mnist", "cifar10", "breastmnist"]
COLORS = {m: plt.cm.tab20(i) for i, m in enumerate(["QK-IQP", "QF-Product", "QF-Ring", "VQC-DR", "Yomo-Matched", "PdrQC-Matched", "C-LR", "C-MLP", "C-Poly", "C-RBF", "C-RFF", "C-Trig", "C-ResNet", "C-CNN"])}


def save(fig: plt.Figure, stem: str, source: pd.DataFrame) -> None:
    source.to_csv(ROOT / "figures/source_data" / f"{stem}.csv", index=False)
    for ext, options in FORMATS.items(): fig.savefig(ROOT / "figures" / ext / f"{stem}.{ext}", bbox_inches="tight", **options)
    plt.close(fig)
    svg = ROOT / "figures/svg" / f"{stem}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


def make_paper_fig2() -> None:
    """Create the paper-specific finite-shot performance figure from saved CIs."""
    datasets = ["fashion_mnist", "cifar10", "breastmnist"]
    methods = ["PdrQC-Matched", "QF-Product", "QF-Ring", "VQC-DR", "Yomo-Matched"]
    budgets = [1, 8, 32, 128, 512]
    finite = pd.read_csv(ROOT / "results/aggregates/finite_shot_performance.csv")
    intervals = pd.read_csv(ROOT / "results/statistics/bootstrap_intervals.csv")
    finite = finite[finite.dataset.isin(datasets) & finite.method.isin(methods)].copy()
    intervals = intervals[
        (intervals.analysis == "primary")
        & (intervals.setting == "finite_shot")
        & intervals.dataset.isin(datasets)
        & intervals.method.isin(methods)
    ].copy()
    intervals["nominal_budget"] = pd.to_numeric(intervals.shot_budget).astype(int)
    source = finite.merge(
        intervals[["dataset", "method", "nominal_budget", "metric", "point_estimate", "ci_95_lower", "ci_95_upper", "replicates", "test_sampling", "seed_hierarchy"]],
        on=["dataset", "method", "nominal_budget"],
        how="inner",
        validate="one_to_one",
    )
    if len(source) != 72 or not np.allclose(source.primary_metric_mean, source.point_estimate, atol=1e-12, rtol=0):
        raise RuntimeError("paper Figure 2 inputs do not match the saved finite-shot intervals")
    if set(source.budget_definition) != {"total_state_preparations_per_image"}:
        raise RuntimeError("paper Figure 2 contains a non-comparable budget definition")
    source["dataset_order"] = pd.Categorical(source.dataset, datasets, ordered=True)
    source["method_order"] = pd.Categorical(source.method, methods, ordered=True)
    source = source.sort_values(["dataset_order", "method_order", "nominal_budget"]).drop(columns=["dataset_order", "method_order"])

    palette = {
        "PdrQC-Matched": "#0072B2",
        "QF-Product": "#E69F00",
        "QF-Ring": "#009E73",
        "VQC-DR": "#CC79A7",
        "Yomo-Matched": "#D55E00",
    }
    markers = {"PdrQC-Matched": "o", "QF-Product": "^", "QF-Ring": "s", "VQC-DR": "D", "Yomo-Matched": "v"}
    panels = [
        ("fashion_mnist", "(a)", "Fashion-MNIST", "Test accuracy (%)", (10, 80), np.arange(10, 81, 10)),
        ("cifar10", "(b)", "CIFAR-10", "Test accuracy (%)", (10, 70), np.arange(10, 71, 10)),
        ("breastmnist", "(c)", "BreastMNIST", "Test AUROC (%)", (35, 90), np.arange(35, 91, 5)),
    ]
    positions = {budget: index for index, budget in enumerate(budgets)}
    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "svg.hashsalt": "real_image_qml_benchmark_2026"})
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 3.15), facecolor="white")
    for axis, (dataset, panel, title, ylabel, limits, ticks) in zip(axes, panels):
        axis.set_facecolor("white")
        for method in methods:
            rows = source[(source.dataset == dataset) & (source.method == method)].sort_values("nominal_budget")
            x = np.asarray([positions[int(value)] for value in rows.nominal_budget])
            point = rows.point_estimate.to_numpy(float) * 100
            lower = rows.ci_95_lower.to_numpy(float) * 100
            upper = rows.ci_95_upper.to_numpy(float) * 100
            axis.fill_between(x, lower, upper, color=palette[method], alpha=0.12, linewidth=0, zorder=1)
            axis.plot(
                x,
                point,
                color=palette[method],
                marker=markers[method],
                linewidth=2.5 if method == "PdrQC-Matched" else 1.35,
                markersize=6 if method == "PdrQC-Matched" else 4.5,
                label=method,
                zorder=3 if method == "PdrQC-Matched" else 2,
            )
        axis.text(0, 1.035, rf"$\bf{{{panel}}}$ {title}", transform=axis.transAxes, ha="left", va="bottom", fontsize=9)
        axis.set_xticks(range(len(budgets)), [str(value) for value in budgets])
        axis.set_xlabel("State preparations per image")
        axis.set_ylabel(ylabel)
        axis.set_ylim(*limits)
        axis.set_yticks(ticks)
        axis.grid(axis="y", color="0.88", linewidth=0.65)
        axis.grid(axis="x", visible=False)
        axis.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    handle_by_label = dict(zip(labels, handles))
    fig.legend(
        [handle_by_label[method] for method in methods],
        methods,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        frameon=False,
        fontsize=8,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    fig.subplots_adjust(left=0.065, right=0.995, top=0.91, bottom=0.25, wspace=0.34)
    save(fig, "fig2_finite_shot_performance", source)


def main() -> None:
    for directory in ["png", "pdf", "svg", "source_data"]: (ROOT / "figures" / directory).mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "figure.dpi": 120, "svg.hashsalt": "real_image_qml_benchmark_2026"})
    # 1 workflow
    nodes = ["Image", "Frozen\nResNet-18", "Train-only\nPCA8", "Shared 8D\nangles", "Quantum + classical\nmethods", "Exact + finite-shot\nevaluation", "Scaling", "Uncertainty", "Statistics"]
    source = pd.DataFrame({"order": range(len(nodes)), "node": nodes}); fig, ax = plt.subplots(figsize=(12, 2.2)); ax.axis("off")
    xs = np.linspace(.04, .96, len(nodes));
    for i, (x, label) in enumerate(zip(xs, nodes)):
        ax.text(x, .5, label, ha="center", va="center", transform=ax.transAxes, bbox={"boxstyle": "round,pad=.35", "fc": "#edf4ff", "ec": "#315a8a"})
        if i: ax.annotate("", xy=(x-.045, .5), xytext=(xs[i-1]+.045, .5), xycoords=ax.transAxes, arrowprops={"arrowstyle": "->", "color": "#555"})
    ax.set_title("Controlled real-image QML benchmark workflow"); save(fig, "fig01_benchmark_pipeline", source)

    exact = pd.read_csv(ROOT / "results/tables/table_05_exact_performance.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, dataset in zip(axes.ravel(), DATASETS):
        d = exact[exact.dataset == dataset].sort_values("primary_metric_mean"); errors = np.vstack([d.primary_metric_mean-d.ci_95_lower, d.ci_95_upper-d.primary_metric_mean]); ax.barh(d.method, d.primary_metric_mean, xerr=errors, capsize=2, color=[COLORS[x] for x in d.method]); ax.set_title(dataset); ax.set_xlabel(d.primary_metric.iloc[0]); ax.grid(axis="x", alpha=.2)
    fig.suptitle("Exact controlled-budget performance (full-image references are retained)"); fig.tight_layout(); save(fig, "fig02_exact_performance", exact)

    finite = pd.read_csv(ROOT / "results/aggregates/finite_shot_performance.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, dataset in zip(axes.ravel(), DATASETS):
        for method, d in finite[finite.dataset == dataset].groupby("method"):
            cost = d.nominal_budget.to_numpy(float)
            if method == "QK-IQP":
                acc = pd.read_csv(ROOT / "results/resources/finite_shot_accounting.csv"); lookup = acc[(acc.dataset == dataset) & (acc.method == method)].set_index("nominal_budget").offline_shot_draws
                cost = np.array([lookup[x] for x in d.nominal_budget])
            ax.plot(cost, d.primary_metric_mean, marker="o", label=method, color=COLORS[method]); ax.set_xscale("log"); ax.set_title(dataset); ax.set_xlabel("state preparations / implied kernel cost"); ax.grid(alpha=.2)
    axes[0,0].legend(ncol=2, fontsize=6); fig.tight_layout(); save(fig, "fig03_finite_shot_curves", finite)

    bins = pd.read_csv(ROOT / "results/aggregates/reliability_bins.csv"); representative = ["PdrQC-Matched", "C-RBF", "C-ResNet"]
    selected = bins[(bins.dataset != "mnist") & (bins.setting == "exact") & bins.method.isin(representative)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5));
    for ax, dataset in zip(axes, DATASETS[1:]):
        ax.plot([0,1],[0,1], "--", color="gray")
        for method, d in selected[selected.dataset == dataset].groupby("method"): ax.plot(d.confidence, d.accuracy, marker="o", label=f"{method} (n={int(d['count'].sum())})", color=COLORS[method])
        ax.set(title=dataset, xlabel="confidence", ylabel="accuracy", xlim=(0,1), ylim=(0,1)); ax.legend(fontsize=6); ax.grid(alpha=.2)
    fig.tight_layout(); save(fig, "fig04_calibration", selected)

    risk = pd.read_csv(ROOT / "results/aggregates/risk_coverage_curves.csv"); selected_risk = risk[(risk.dataset != "mnist") & (risk.setting == "exact") & risk.method.isin(representative)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5));
    for ax, dataset in zip(axes, DATASETS[1:]):
        for method, d in selected_risk[selected_risk.dataset == dataset].groupby("method"): ax.plot(d.coverage, d.risk, label=method, color=COLORS[method])
        ax.set(title=dataset, xlabel="coverage", ylabel="risk"); ax.legend(fontsize=6); ax.grid(alpha=.2)
    fig.tight_layout(); save(fig, "fig05_risk_coverage", selected_risk)

    account = pd.read_csv(ROOT / "results/resources/finite_shot_accounting.csv"); pareto = finite.merge(account[["dataset", "method", "nominal_budget", "offline_shot_draws"]], on=["dataset", "method", "nominal_budget"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4));
    for method, d in pareto.groupby("method"):
        axes[0].scatter(d.actual_state_preparations_per_image_mean, d.primary_metric_mean, label=method, color=COLORS[method], s=15); axes[1].scatter(d.offline_shot_draws, d.primary_metric_mean, color=COLORS[method], s=15)
    for ax, label in zip(axes, ["state preparations per image", "total offline shot draws"]): ax.set(xlabel=label, ylabel="primary metric", xscale="log"); ax.grid(alpha=.2)
    axes[0].legend(fontsize=6); fig.tight_layout(); save(fig, "fig06_performance_resource_pareto", pareto)

    uncertainty = pd.read_csv(ROOT / "results/aggregates/uncertainty_metrics.csv"); uq = uncertainty[uncertainty.setting == "finite_shot"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4));
    for method, d in uq.groupby("method"):
        g = d.groupby("shot_budget", as_index=False).agg(model_uncertainty=("model_uncertainty", "mean"), shot_uncertainty=("shot_uncertainty", "mean")); g.shot_budget = pd.to_numeric(g.shot_budget)
        axes[0].plot(g.shot_budget, g.model_uncertainty, marker="o", label=method, color=COLORS[method]); axes[1].plot(g.shot_budget, g.shot_uncertainty, marker="o", label=method, color=COLORS[method])
    for ax, title in zip(axes, ["Model uncertainty", "Quantum-measurement uncertainty"]): ax.set(xscale="log", xlabel="shot budget", ylabel="mean squared probability deviation", title=title); ax.grid(alpha=.2)
    axes[0].legend(fontsize=6); fig.tight_layout(); save(fig, "fig07_uncertainty_decomposition", uq)

    rank = exact.copy(); rank["rank"] = rank.groupby("dataset").primary_metric_mean.rank(ascending=False); pivot = rank.pivot(index="method", columns="dataset", values="rank").reindex(columns=DATASETS)
    fig, ax = plt.subplots(figsize=(7, 6)); im = ax.imshow(pivot, cmap="viridis_r", aspect="auto"); ax.set_xticks(range(4), DATASETS, rotation=25); ax.set_yticks(range(len(pivot)), pivot.index)
    for i in range(len(pivot)):
        for j in range(4): ax.text(j, i, f"{pivot.iloc[i,j]:.0f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="rank (1=best)"); fig.tight_layout(); save(fig, "fig08_method_ranking_heatmap", rank[["dataset", "method", "primary_metric_mean", "rank"]])

    resource_rows = pd.read_csv(ROOT / "results/tables/table_10_resources.csv"); resource = resource_rows.groupby("method", as_index=False).agg(depth=("depth","max"), two_qubit_gates=("two_qubit_gates","max"), measurement_groups=("measurement_groups","max"), state_preparations=("actual_state_preparations_per_image","max"))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7));
    for ax, column in zip(axes.ravel(), ["depth", "two_qubit_gates", "measurement_groups", "state_preparations"]): ax.bar(resource.method, resource[column], color=[COLORS[x] for x in resource.method]); ax.set_title(column.replace("_", " ")); ax.tick_params(axis="x", rotation=45); ax.set_yscale("log" if column == "state_preparations" else "linear")
    fig.tight_layout(); save(fig, "fig09_circuit_measurement_resources", resource)

    scaling = pd.read_csv(ROOT / "results/tables/table_12_data_scaling.csv"); size = {"D_1k":1000,"D_5k":5000,"D_10k":10000,"D_full":55000,"D_25pct":136,"D_50pct":273,"D_100pct":546}; scaling["training_size_numeric"] = scaling.training_scale.map(size)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8));
    for ax, dataset in zip(axes.ravel(), DATASETS):
        for method, d in scaling[scaling.dataset == dataset].groupby("method"):
            ax.errorbar(d.training_size_numeric, d.primary_metric_mean, yerr=np.vstack([d.primary_metric_mean-d.ci_95_lower, d.ci_95_upper-d.primary_metric_mean]), marker="o", label=method, color=COLORS[method], capsize=2)
        boundaries = pd.read_csv(ROOT / "results/aggregates/data_scaling_boundaries.csv"); bd = boundaries[boundaries.dataset == dataset]
        for item in bd.itertuples(): ax.scatter(size[item.training_scale], ax.get_ylim()[0], marker="x", color=COLORS[item.method], s=35)
        ax.set(title=dataset, xlabel="training size", ylabel="primary metric", xscale="log"); ax.grid(alpha=.2)
    axes[0,0].legend(ncol=2, fontsize=5); fig.tight_layout(); save(fig, "fig10_learning_curves", scaling)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4));
    for method, d in scaling.groupby("method"):
        axes[0].scatter(d.training_size_numeric, d.primary_metric_mean, color=COLORS[method], s=10, label=method); axes[2].scatter(d.training_size_numeric, d.kernel_entries.replace(0, np.nan), color=COLORS[method], s=10)
    exact_raw = pd.read_csv(ROOT / "results/raw/data_scaling_runs.csv"); exact_raw["training_size_numeric"] = exact_raw.training_scale.map(size); exact_raw["wall_seconds"] = pd.to_numeric(exact_raw.wall_seconds, errors="coerce")
    axes[1].scatter(exact_raw.training_size_numeric, exact_raw.wall_seconds, s=8, alpha=.5)
    axes[0].set(xlabel="training size", ylabel="primary metric", xscale="log"); axes[1].set(xlabel="training size", ylabel="inference wall seconds", xscale="log", yscale="log"); axes[2].set(xlabel="training size", ylabel="kernel entries", xscale="log", yscale="log")
    axes[0].legend(fontsize=5, ncol=2); [ax.grid(alpha=.2) for ax in axes]; fig.tight_layout(); source11 = scaling.merge(exact_raw.groupby(["dataset", "training_scale", "method"], as_index=False).wall_seconds.mean(), on=["dataset", "training_scale", "method"], how="left"); save(fig, "fig11_data_scaling_cost", source11)

    stability = pd.read_csv(ROOT / "results/statistics/data_scaling_intervals.csv"); labels = sorted(set(stability.scale_A) | set(stability.scale_B)); matrix = np.full((len(labels), len(labels)), np.nan); counts = np.zeros_like(matrix)
    for row in stability.itertuples(): i, j = labels.index(row.scale_A), labels.index(row.scale_B); matrix[i,j]=matrix[j,i]=row.spearman_rho; counts[i,j]=counts[j,i]=row.common_methods
    np.fill_diagonal(matrix, 1); fig, ax = plt.subplots(figsize=(7, 6)); im=ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm"); ax.set_xticks(range(len(labels)), labels, rotation=35); ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if np.isfinite(matrix[i,j]): ax.text(j,i,f"{matrix[i,j]:.2f}\nn={int(counts[i,j]) if i!=j else '-'}",ha="center",va="center",fontsize=6)
    fig.colorbar(im, ax=ax, label="Spearman rho"); fig.tight_layout(); save(fig, "fig12_ranking_stability", stability)
    print({"figures": 12, "png": len(list((ROOT/'figures/png').glob('*.png'))), "pdf": len(list((ROOT/'figures/pdf').glob('*.pdf'))), "svg": len(list((ROOT/'figures/svg').glob('*.svg')))})


if __name__ == "__main__":
    import sys

    make_paper_fig2() if "--paper-fig2-only" in sys.argv else main()
