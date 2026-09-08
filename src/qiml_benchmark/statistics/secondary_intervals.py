"""Add exact ECE, error-detection AUROC, and AURC bootstrap intervals."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from qiml_benchmark.metrics.core import classification_metrics
from qiml_benchmark.statistics.run_all import B, ROOT, SEED, load_stack, selections, stratified_indices


def run_distribution(labels: np.ndarray, probability: np.ndarray, index: np.ndarray) -> dict[str, np.ndarray]:
    output = {name: np.empty(B) for name in ["ece_15", "error_detection_auroc", "aurc"]}; n = len(labels)
    for start in range(0, B, 25):
        stop = min(B, start + 25); ix = index[start:stop]; truth = labels[ix]; prob = probability[ix]; confidence = prob.max(2); prediction = prob.argmax(2); correct = prediction == truth; error = ~correct
        order = np.argsort(confidence, axis=1, kind="stable"); ordered_conf = np.take_along_axis(confidence, order, axis=1); ordered_correct = np.take_along_axis(correct, order, axis=1)
        ece = np.zeros(stop-start)
        for group in np.array_split(np.arange(n), 15): ece += len(group)/n * np.abs(ordered_correct[:, group].mean(1)-ordered_conf[:, group].mean(1))
        output["ece_15"][start:stop] = ece
        uncertainty = 1-confidence; ranks = rankdata(uncertainty, axis=1, method="average"); positives = error.sum(1); negatives = n-positives; numerator=(ranks*error).sum(1)-positives*(positives+1)/2
        output["error_detection_auroc"][start:stop] = np.divide(numerator, positives*negatives, out=np.full(stop-start,np.nan), where=(positives*negatives)!=0)
        reverse = order[:, ::-1]; ordered_error = np.take_along_axis(error, reverse, axis=1); risk=np.cumsum(ordered_error,axis=1)/np.arange(1,n+1); output["aurc"][start:stop]=np.trapezoid(risk,np.arange(1,n+1)/n,axis=1)
    return output


def worker(args: tuple[str, str, list[str]]) -> tuple[list[dict], dict[str, np.ndarray]]:
    dataset, method, files = args; frame = pd.DataFrame({"prediction_file": files, "model_seed": [Path(x).stem.rsplit("_",1)[1] for x in files], "measurement_seed": 0})
    labels, stack, thresholds = load_stack(frame); index = stratified_indices(labels); flat=stack.reshape(-1,len(labels),stack.shape[-1]); run_dist=[run_distribution(labels,p,index) for p in flat]; choice=selections(stack.shape[0],stack.shape[1],SEED+sum(map(ord,dataset+"exact"+"exact")))
    rows=[]; distributions={}
    raw = pd.read_csv(ROOT / "results/raw/exact_test_runs.csv"); group=raw[(raw.dataset==dataset)&(raw.method==method)]
    for metric in ["ece_15","error_detection_auroc","aurc"]:
        matrix=np.stack([x[metric] for x in run_dist]); dist=np.take_along_axis(matrix.T,choice,axis=1).mean(1); lo,hi=np.nanquantile(dist,[.025,.975]); point=pd.to_numeric(group[f"metric_{metric}"],errors="coerce").mean()
        rows.append({"analysis":"uncertainty_secondary","dataset":dataset,"method":method,"setting":"exact","shot_budget":"exact","metric":metric,"point_estimate":point,"ci_95_lower":lo,"ci_95_upper":hi,"replicates":B,"test_sampling":"stratified_image_bootstrap","seed_hierarchy":"model_then_measurement"})
        distributions[f"exact|{dataset}|{method}|exact|{metric}"]=dist
    return rows,distributions


def main() -> None:
    raw=pd.read_csv(ROOT/"results/raw/exact_test_runs.csv"); tasks=[]
    for (dataset,method),group in raw.groupby(["dataset","method"],sort=True): tasks.append((dataset,method,group.prediction_file.tolist()))
    rows=[]; extra={}
    with ProcessPoolExecutor(max_workers=4) as pool:
        for produced,distributions in pool.map(worker,tasks): rows.extend(produced); extra.update(distributions)
    path=ROOT/"results/statistics/bootstrap_intervals.csv"; frame=pd.read_csv(path); frame=frame[frame.analysis!="uncertainty_secondary"]; pd.concat([frame,pd.DataFrame(rows)],ignore_index=True).to_csv(path,index=False)
    archive_path=ROOT/"results/statistics/resampling_distributions.npz"; existing=dict(np.load(archive_path)); existing.update({hashlib.sha256(k.encode()).hexdigest()[:20]:v for k,v in extra.items()}); np.savez_compressed(archive_path,**existing)
    meta_path=ROOT/"results/statistics/resampling_metadata.json"; metadata=json.loads(meta_path.read_text()); metadata["distribution_archive_sha256"]=hashlib.sha256(archive_path.read_bytes()).hexdigest(); metadata["exact_secondary_uncertainty_intervals"]=len(rows); meta_path.write_text(json.dumps(metadata,indent=2,sort_keys=True)+"\n")
    print({"additional_intervals":len(rows),"total_intervals":len(frame)+len(rows),"archive_arrays":len(existing)})


if __name__ == "__main__": main()
