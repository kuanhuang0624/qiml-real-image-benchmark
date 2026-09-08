"""Frozen stratified bootstrap and paired randomization analyses."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from qiml_benchmark.paths import workspace_root
ROOT = workspace_root()
B = 10_000
P = 10_000
SEED = 8675309
PRIMARY = {"breastmnist": "auroc", "mnist": "accuracy", "fashion_mnist": "accuracy", "cifar10": "accuracy"}


def stratified_indices(labels: np.ndarray, replicates: int = B) -> np.ndarray:
    rng = np.random.default_rng(SEED + len(labels) + int(labels.sum()))
    chunks = []
    for value in np.unique(labels):
        positions = np.flatnonzero(labels == value)
        chunks.append(rng.choice(positions, size=(replicates, len(positions)), replace=True).astype(np.int32))
    return np.concatenate(chunks, axis=1)


def load_stack(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group = group.copy()
    group["_m"] = group.model_seed.fillna(42).astype(str)
    if "measurement_seed" not in group: group["measurement_seed"] = 0
    group["_r"] = group.measurement_seed.fillna(0).astype(int)
    models = sorted(group._m.unique()); measurements = sorted(group._r.unique())
    first = np.load(ROOT / group.iloc[0].prediction_file); labels = first["label"]
    arrays, thresholds = [], []
    for model in models:
        model_arrays, model_thresholds = [], []
        subset = group[group._m == model]
        available = sorted(subset._r.unique())
        for measurement in available:
            row = subset[subset._r == measurement].iloc[0]; z = np.load(ROOT / row.prediction_file)
            if not np.array_equal(labels, z["label"]): raise ValueError("prediction label ordering mismatch")
            model_arrays.append(z["calibrated_probability"]); model_thresholds.append(float(z["threshold"]))
        arrays.append(model_arrays); thresholds.append(model_thresholds)
    # Frozen conditions are rectangular in model and measurement seeds.
    return labels, np.asarray(arrays), np.asarray(thresholds)


def selections(m: int, r: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed); out = np.empty((B, m * r), dtype=np.int16)
    for b in range(B):
        chosen_m = rng.integers(0, m, size=m); flat = []
        for mm in chosen_m:
            chosen_r = rng.integers(0, r, size=r); flat.extend(mm * r + chosen_r)
        out[b] = flat
    return out


def point_metric(labels: np.ndarray, prob: np.ndarray, threshold: float, metric: str) -> float:
    if metric == "accuracy":
        pred = (prob[:, 1] >= threshold).astype(int) if np.isfinite(threshold) else prob.argmax(1)
        return float(np.mean(pred == labels))
    if metric == "auroc": return binary_auc(labels, prob[:, 1])
    if metric == "auprc": return binary_ap(labels, prob[:, 1])
    if metric == "macro_f1":
        pred = (prob[:, 1] >= threshold).astype(int) if np.isfinite(threshold) else prob.argmax(1)
        classes = prob.shape[1]; matrix = np.bincount(labels * classes + pred, minlength=classes * classes).reshape(classes, classes)
        tp = np.diag(matrix); denominator = matrix.sum(0) + matrix.sum(1)
        return float(np.divide(2 * tp, denominator, out=np.zeros(classes, dtype=float), where=denominator != 0).mean())
    onehot = np.eye(prob.shape[1])[labels]
    if metric == "nll": return float(np.mean(-np.log(np.clip(prob[np.arange(len(labels)), labels], 1e-12, 1))))
    if metric == "brier": return float(np.mean(np.sum((prob - onehot) ** 2, axis=1)))
    raise KeyError(metric)


def binary_auc(labels: np.ndarray, score: np.ndarray) -> float:
    labels = np.asarray(labels, bool); positives = int(labels.sum()); negatives = len(labels) - positives
    if not positives or not negatives: return np.nan
    ranks = rankdata(score, method="average")
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def binary_ap(labels: np.ndarray, score: np.ndarray) -> float:
    labels = np.asarray(labels, bool); positives = int(labels.sum())
    if not positives: return np.nan
    ordered = labels[np.argsort(-score, kind="stable")]
    return float(np.sum((np.cumsum(ordered) / np.arange(1, len(labels) + 1)) * ordered) / positives)


def bootstrap_metric(labels: np.ndarray, stack: np.ndarray, thresholds: np.ndarray, metric: str, index: np.ndarray, pair_seed: int) -> np.ndarray:
    m, r, n, _ = stack.shape; flat = stack.reshape(m * r, n, -1); threshold = thresholds.reshape(-1)
    choice = selections(m, r, pair_seed); output = np.empty(B)
    if metric in {"accuracy", "nll", "brier"}:
        values = []
        for run, prob in enumerate(flat):
            if metric == "accuracy":
                pred = (prob[:, 1] >= threshold[run]).astype(int) if np.isfinite(threshold[run]) else prob.argmax(1)
                values.append((pred == labels).astype(np.float32))
            elif metric == "nll": values.append(-np.log(np.clip(prob[np.arange(n), labels], 1e-12, 1)).astype(np.float32))
            else: values.append(np.sum((prob - np.eye(prob.shape[1])[labels]) ** 2, axis=1).astype(np.float32))
        values = np.asarray(values)
        for start in range(0, B, 50):
            stop = min(B, start + 50); run_mean = values[choice[start:stop]].mean(axis=1)
            output[start:stop] = np.take_along_axis(run_mean, index[start:stop], axis=1).mean(axis=1)
        return output
    if metric == "auprc":
        for start in range(0, B, 100):
            stop = min(B, start + 100); ix = index[start:stop]; sampled_labels = labels[ix]
            selected = np.take_along_axis(flat[:, :, 1][choice[start:stop]], ix[:, None, :], axis=2)
            order = np.argsort(-selected, axis=2, kind="stable"); ordered_labels = np.take_along_axis(np.broadcast_to(sampled_labels[:, None, :], selected.shape), order, axis=2)
            precision = np.cumsum(ordered_labels, axis=2) / np.arange(1, len(labels) + 1)
            positives = ordered_labels.sum(2); ap = np.divide((precision * ordered_labels).sum(2), positives, out=np.full(positives.shape, np.nan), where=positives != 0)
            output[start:stop] = np.nanmean(ap, axis=1)
        return output
    if metric == "auroc":
        for start in range(0, B, 50):
            stop = min(B, start + 50); ix = index[start:stop]; sampled_labels = labels[ix]
            selected = np.take_along_axis(flat[:, :, 1][choice[start:stop]], ix[:, None, :], axis=2)
            ranks = rankdata(selected, axis=2, method="average"); positive = sampled_labels[:, None, :].astype(bool)
            positives = positive.sum(2); negatives = len(labels) - positives
            numerator = (ranks * positive).sum(2) - positives * (positives + 1) / 2; denominator = positives * negatives
            auc = np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan), where=denominator != 0)
            output[start:stop] = np.nanmean(auc, axis=1)
        return output
    if metric == "macro_f1":
        predictions = []
        for run, prob in enumerate(flat):
            predictions.append((prob[:, 1] >= threshold[run]).astype(np.int16) if np.isfinite(threshold[run]) else prob.argmax(1).astype(np.int16))
        predictions = np.asarray(predictions); classes = stack.shape[-1]
        for start in range(0, B, 100):
            stop = min(B, start + 100); ix = index[start:stop]; selected = np.take_along_axis(predictions[choice[start:stop]], ix[:, None, :], axis=2); truth = labels[ix][:, None, :]
            count, runs = selected.shape[:2]; codes = truth * classes + selected
            offsets = np.arange(count * runs).reshape(count, runs, 1) * classes * classes
            matrix = np.bincount((codes + offsets).ravel(), minlength=count * runs * classes * classes).reshape(count, runs, classes, classes)
            tp = np.diagonal(matrix, axis1=2, axis2=3); denominator = matrix.sum(2) + matrix.sum(3)
            f1 = np.divide(2 * tp, denominator, out=np.zeros(tp.shape, dtype=float), where=denominator != 0)
            output[start:stop] = f1.mean(2).mean(1)
        return output
    for b in range(B):
        ix = index[b]; vals = [point_metric(labels[ix], flat[j][ix], threshold[j], metric) for j in choice[b]]
        output[b] = np.mean(vals)
    return output


def ci_row(kind: str, dataset: str, method: str, setting: str, budget: str, metric: str, point: float, dist: np.ndarray) -> dict:
    lo, hi = np.quantile(dist, [.025, .975])
    return {"analysis": kind, "dataset": dataset, "method": method, "setting": setting, "shot_budget": budget,
            "metric": metric, "point_estimate": point, "ci_95_lower": lo, "ci_95_upper": hi, "replicates": B,
            "test_sampling": "stratified_image_bootstrap", "seed_hierarchy": "model_then_measurement"}


def holm(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues); adjusted = np.empty(len(pvalues)); running = 0.0; count = len(pvalues)
    for rank, idx in enumerate(order):
        running = max(running, (count - rank) * pvalues[idx]); adjusted[idx] = min(1.0, running)
    return adjusted


def aurc(labels: np.ndarray, prob: np.ndarray) -> float:
    uncertainty = 1 - prob.max(1); errors = prob.argmax(1) != labels; order = np.argsort(uncertainty)
    risk = np.cumsum(errors[order]) / np.arange(1, len(labels) + 1)
    return float(np.trapezoid(risk, np.arange(1, len(labels) + 1) / len(labels)))


def permutation_test(labels: np.ndarray, a: np.ndarray, b: np.ndarray, ta: np.ndarray, tb: np.ndarray, metric: str, seed: int) -> tuple[float, float, np.ndarray]:
    af, bf = a.reshape(-1, len(labels), a.shape[-1]), b.reshape(-1, len(labels), b.shape[-1])
    if len(af) != len(bf): raise ValueError("paired stacks must contain equal run counts")
    if metric == "accuracy":
        ca = np.stack([((p[:, 1] >= t).astype(int) if np.isfinite(t) else p.argmax(1)) == labels for p, t in zip(af, ta.reshape(-1))]).mean(0)
        cb = np.stack([((p[:, 1] >= t).astype(int) if np.isfinite(t) else p.argmax(1)) == labels for p, t in zip(bf, tb.reshape(-1))]).mean(0)
        delta = ca - cb; observed = float(delta.mean()); rng = np.random.default_rng(seed); dist = np.empty(P)
        for start in range(0, P, 100):
            signs = rng.choice(np.array([-1., 1.]), size=(min(100, P-start), len(labels)))
            dist[start:start+len(signs)] = (signs * delta).mean(1)
    elif metric == "brier":
        onehot = np.eye(af.shape[-1])[labels]
        delta = np.sum((af - onehot) ** 2, axis=2).mean(0) - np.sum((bf - onehot) ** 2, axis=2).mean(0)
        observed = float(delta.mean()); rng = np.random.default_rng(seed); dist = np.empty(P)
        for start in range(0, P, 100):
            signs = rng.choice(np.array([-1., 1.]), size=(min(100, P-start), len(labels)))
            dist[start:start+len(signs)] = (signs * delta).mean(1)
    else:
        rng = np.random.default_rng(seed); dist = np.empty(P)
        def score(x: np.ndarray) -> float:
            if metric == "auroc": return float(np.mean([binary_auc(labels, p[:, 1]) for p in x]))
            return aurc(labels, x.mean(0))
        observed = score(af) - score(bf)
        if metric == "auroc":
            positive = labels.astype(bool); pair_mask = positive[:, None] & (~positive[None, :]); denominator = pair_mask.sum()
            for start in range(0, P, 50):
                count = min(50, P - start); swap = rng.integers(0, 2, size=(count, len(labels)), dtype=np.int8).astype(bool)
                ap = np.where(swap[:, None, :], bf[None, :, :, 1], af[None, :, :, 1]); bp = np.where(swap[:, None, :], af[None, :, :, 1], bf[None, :, :, 1])
                ad = ap[:, :, :, None] - ap[:, :, None, :]; bd = bp[:, :, :, None] - bp[:, :, None, :]
                auc_a = (((ad > 0) + .5 * (ad == 0)) * pair_mask).sum(axis=(2, 3)) / denominator
                auc_b = (((bd > 0) + .5 * (bd == 0)) * pair_mask).sum(axis=(2, 3)) / denominator
                dist[start:start + count] = auc_a.mean(1) - auc_b.mean(1)
        else:
            for k in range(P):
                swap = rng.integers(0, 2, len(labels), dtype=np.int8).astype(bool)
                ap = np.where(swap[None, :, None], bf, af); bp = np.where(swap[None, :, None], af, bf)
                dist[k] = score(ap) - score(bp)
    pvalue = (1 + np.sum(np.abs(dist) >= abs(observed))) / (P + 1)
    return observed, float(pvalue), dist


def main() -> None:
    stats = ROOT / "results/statistics"; stats.mkdir(parents=True, exist_ok=True)
    exact = pd.read_csv(ROOT / "results/raw/exact_test_runs.csv"); finite = pd.read_csv(ROOT / "results/raw/finite_shot_runs.csv"); scaling = pd.read_csv(ROOT / "results/raw/data_scaling_runs.csv")
    labels_by_dataset, index_by_dataset = {}, {}
    for dataset, group in exact[exact.status == "COMPLETED"].groupby("dataset"):
        labels, _, _ = load_stack(group.iloc[[0]]); labels_by_dataset[dataset] = labels; index_by_dataset[dataset] = stratified_indices(labels)
    intervals, distributions, primary_dist = [], {}, {}

    sources = [("exact", exact[exact.status == "COMPLETED"], ["dataset", "method"]),
               ("finite_shot", finite[finite.status == "COMPLETED"], ["dataset", "method", "nominal_budget"]),
               ("data_scaling", scaling[scaling.status == "COMPLETED"], ["dataset", "training_scale", "method"])]
    for setting, frame, keys in sources:
        for key, group in frame.groupby(keys, sort=True):
            if not isinstance(key, tuple): key = (key,)
            dataset = key[0]; method = key[-1] if setting != "finite_shot" else key[1]
            budget = "exact" if setting == "exact" else (str(int(key[2])) if setting == "finite_shot" else str(key[1]))
            labels, stack, thresholds = load_stack(group); metric = PRIMARY[dataset]
            pair_seed = SEED + sum(map(ord, dataset + setting + budget))
            dist = bootstrap_metric(labels, stack, thresholds, metric, index_by_dataset[dataset], pair_seed)
            points = [point_metric(labels, p, t, metric) for p, t in zip(stack.reshape(-1, len(labels), stack.shape[-1]), thresholds.reshape(-1))]
            token = f"{setting}|{dataset}|{method}|{budget}|{metric}"; distributions[token] = dist; primary_dist[(setting, dataset, method, budget)] = dist
            intervals.append(ci_row("primary", dataset, method, setting, budget, metric, float(np.mean(points)), dist))
            if setting == "exact":
                secondary = "auprc" if dataset == "breastmnist" else "macro_f1"
                for extra in [secondary, "nll", "brier"]:
                    extra_dist = bootstrap_metric(labels, stack, thresholds, extra, index_by_dataset[dataset], pair_seed)
                    pts = [point_metric(labels, p, t, extra) for p, t in zip(stack.reshape(-1, len(labels), stack.shape[-1]), thresholds.reshape(-1))]
                    distributions[f"exact|{dataset}|{method}|exact|{extra}"] = extra_dist
                    intervals.append(ci_row("secondary", dataset, method, setting, budget, extra, float(np.mean(pts)), extra_dist))
    pd.DataFrame(intervals).to_csv(stats / "bootstrap_intervals.csv", index=False)

    # Frozen primary comparisons.
    contrasts = json.loads((ROOT / "configs/frozen/PRIMARY_CONTRASTS.yaml").read_text())["contrasts"]
    test_rows, perm_distributions = [], {}
    for number, contrast in enumerate(contrasts):
        dataset, setting = contrast["dataset"], contrast["setting"]; metric = contrast["metric"].lower()
        if setting == "exact": source = exact; budget = "exact"; filt = (source.dataset == dataset)
        else:
            budget = str(int(setting.rsplit("_", 1)[1])); source = finite; filt = (source.dataset == dataset) & (source.nominal_budget == int(budget))
        ga = source[filt & (source.method == contrast["method_A"])] ; gb = source[filt & (source.method == contrast["method_B"])]
        labels, a, ta = load_stack(ga); labels_b, b, tb = load_stack(gb)
        if not np.array_equal(labels, labels_b): raise ValueError("paired labels differ")
        observed, raw_p, perm = permutation_test(labels, a, b, ta, tb, metric, SEED + number)
        da = primary_dist[("exact" if setting == "exact" else "finite_shot", dataset, contrast["method_A"], budget)]
        db = primary_dist[("exact" if setting == "exact" else "finite_shot", dataset, contrast["method_B"], budget)]
        paired = da - db; lo, hi = np.quantile(paired, [.025, .975]); perm_distributions[contrast["id"]] = perm
        test_rows.append({"contrast": contrast["id"], "dataset": dataset, "metric": metric, "method_A": contrast["method_A"], "method_B": contrast["method_B"],
                          "setting": setting, "effect_A_minus_B": observed, "ci_95_lower": lo, "ci_95_upper": hi, "raw_p": raw_p, "permutations": P, "status": "ESTIMATED"})
    primary_tests = pd.DataFrame(test_rows); primary_tests["holm_p"] = holm(primary_tests.raw_p.to_numpy())
    primary_tests["interpretation"] = np.where((primary_tests.ci_95_lower <= 0) & (primary_tests.ci_95_upper >= 0), "No statistically detectable difference was observed.", np.where(primary_tests.holm_p < .05, "Statistically detectable paired difference.", "No statistically detectable difference was observed."))
    primary_tests.to_csv(stats / "primary_tests.csv", index=False)

    # Six separately corrected calibration/selective-risk tests use validation-selected H1 pairs.
    uncertainty_rows = []
    for number, dataset in enumerate(["fashion_mnist", "cifar10", "breastmnist"]):
        h1 = next(x for x in contrasts if x["id"] == f"H1_{dataset}")
        ga = exact[(exact.dataset == dataset) & (exact.method == h1["method_A"])]; gb = exact[(exact.dataset == dataset) & (exact.method == h1["method_B"])]
        labels, a, ta = load_stack(ga); _, b, tb = load_stack(gb)
        for metric in ["brier", "aurc"]:
            observed, raw_p, perm = permutation_test(labels, a, b, ta, tb, metric, SEED + 100 + number * 2 + (metric == "aurc"))
            # Paired stratified image bootstrap; exact H1 methods each have one run.
            ix = index_by_dataset[dataset]; boot = np.empty(B)
            af, bf = a.reshape(-1, len(labels), a.shape[-1]), b.reshape(-1, len(labels), b.shape[-1])
            for k in range(B):
                sample = ix[k]
                if metric == "brier":
                    oh = np.eye(a.shape[-1])[labels[sample]]; boot[k] = np.mean(np.sum((af[:, sample] - oh) ** 2, axis=2)) - np.mean(np.sum((bf[:, sample] - oh) ** 2, axis=2))
                else: boot[k] = aurc(labels[sample], af[:, sample].mean(0)) - aurc(labels[sample], bf[:, sample].mean(0))
            lo, hi = np.quantile(boot, [.025, .975]); key = f"U_{metric}_{dataset}"; perm_distributions[key] = perm; distributions[key] = boot
            uncertainty_rows.append({"contrast": key, "dataset": dataset, "metric": metric, "method_A": h1["method_A"], "method_B": h1["method_B"], "effect_A_minus_B": observed, "ci_95_lower": lo, "ci_95_upper": hi, "raw_p": raw_p, "permutations": P, "status": "ESTIMATED"})
    uncertainty_tests = pd.DataFrame(uncertainty_rows); uncertainty_tests["holm_p"] = holm(uncertainty_tests.raw_p.to_numpy())
    uncertainty_tests["interpretation"] = np.where((uncertainty_tests.ci_95_lower <= 0) & (uncertainty_tests.ci_95_upper >= 0), "No statistically detectable difference was observed.", np.where(uncertainty_tests.holm_p < .05, "Statistically detectable paired difference.", "No statistically detectable difference was observed."))
    uncertainty_tests.to_csv(stats / "uncertainty_tests.csv", index=False)
    pd.concat([primary_tests.assign(family="primary_12"), uncertainty_tests.assign(family="uncertainty_6")], ignore_index=True).to_csv(stats / "holm_adjustments.csv", index=False)

    # Scaling rank-correlation confidence intervals from common completed methods.
    scaling_agg = pd.read_csv(ROOT / "results/aggregates/data_scaling_performance.csv"); stability = pd.read_csv(ROOT / "results/aggregates/ranking_stability.csv"); rank_rows = []
    for row in stability.itertuples():
        left = scaling_agg[(scaling_agg.dataset == row.dataset) & (scaling_agg.training_scale == row.scale_A)].method
        right = scaling_agg[(scaling_agg.dataset == row.dataset) & (scaling_agg.training_scale == row.scale_B)].method
        common = sorted(set(left) & set(right))
        matrix_a = np.column_stack([primary_dist[("data_scaling", row.dataset, method, row.scale_A)] for method in common])
        matrix_b = np.column_stack([primary_dist[("data_scaling", row.dataset, method, row.scale_B)] for method in common])
        rank_a = np.argsort(np.argsort(-matrix_a, axis=1), axis=1).astype(float)
        rank_b = np.argsort(np.argsort(-matrix_b, axis=1), axis=1).astype(float)
        rank_a -= rank_a.mean(1, keepdims=True); rank_b -= rank_b.mean(1, keepdims=True)
        values = np.sum(rank_a * rank_b, axis=1) / np.sqrt(np.sum(rank_a ** 2, axis=1) * np.sum(rank_b ** 2, axis=1))
        lo, hi = np.nanquantile(values, [.025, .975]); rank_rows.append({"dataset": row.dataset, "scale_A": row.scale_A, "scale_B": row.scale_B, "common_methods": len(common), "spearman_rho": row.spearman_rho, "ci_95_lower": lo, "ci_95_upper": hi, "methods_included": ";".join(common)})
    pd.DataFrame(rank_rows).to_csv(stats / "data_scaling_intervals.csv", index=False)

    raw_path = stats / "resampling_distributions.npz"; np.savez_compressed(raw_path, **{hashlib.sha256(k.encode()).hexdigest()[:20]: v for k, v in {**distributions, **perm_distributions}.items()})
    metadata = {"bootstrap_replicates": B, "permutations": P, "seed": SEED, "stratification": "class", "paired_images": True,
                "hierarchy": ["model_seed", "measurement_seed", "test_image"], "distribution_archive": str(raw_path.relative_to(ROOT)),
                "distribution_archive_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(), "primary_Holm_family": 12, "uncertainty_Holm_family": 6}
    (stats / "resampling_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print({"intervals": len(intervals), "primary_tests": len(primary_tests), "uncertainty_tests": len(uncertainty_tests), "ranking_intervals": len(rank_rows)})


if __name__ == "__main__":
    main()
