"""Multiprocessing helper for replication-level parallelism.

Each worker process runs ONE replication's full counterfactual set (4 scenarios,
shared population/network) and writes its own output files to disk. The master
process orchestrates work distribution and progress reporting.

Why per-replication granularity:
- Within a replication, the 4 scenarios MUST share population/network state for
  the counterfactual interpretation to hold. So we don't parallelize across
  scenarios.
- Across replications, every rep is independent — each has its own seed,
  population, network, and intervention state. Trivially parallel.
- File I/O happens inside the worker, so the master only sees small status
  messages back, not large trajectory arrays.

Resumability is preserved: each worker checks rep file presence on disk before
running, so a re-launch picks up where it left off.
"""

import json
import os
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

from experiments.counterfactual import CounterfactualAnalyzer
from experiments.scenario import build_scenarios


def _save_trajectory_csv(trajectory: dict, path: str):
    if trajectory is None:
        return
    n = len(trajectory['days'])
    data = {
        'day': trajectory['days'][:n],
        'S': trajectory['S'][:n],
        'E': trajectory['E'][:n],
        'I': trajectory['I'][:n],
        'R': trajectory['R'][:n],
        'prevalence': trajectory['prevalence'][:n],
    }
    if 'mode' in trajectory:
        data['mode'] = trajectory['mode'][:n]
    pd.DataFrame(data).to_csv(path, index=False)


def _worker_run_one(args):
    """Run one replication and write its files. Returns (rep, status, metrics_per_scenario)."""
    rep, disease_config, sim_config, output_dir, scenario_names = args

    # Skip if all scenario files already exist (resumability check)
    all_exist = True
    for sc in scenario_names:
        rep_file = os.path.join(output_dir, sc, f"rep_{rep:03d}.json")
        if not os.path.exists(rep_file):
            all_exist = False
            break
    if all_exist:
        return rep, "skipped", {}

    scenarios = build_scenarios(disease_config, sim_config)
    analyzer = CounterfactualAnalyzer(disease_config, sim_config)
    rep_results = analyzer.run_counterfactual_set(rep, scenarios, verbose=False)

    metrics_per_scenario = {}
    for scenario_name, result in rep_results.items():
        metrics = result.get("metrics", {})
        metrics_per_scenario[scenario_name] = metrics

        rep_dir = os.path.join(output_dir, scenario_name)
        os.makedirs(rep_dir, exist_ok=True)

        rep_file = os.path.join(rep_dir, f"rep_{rep:03d}.json")
        save_data = {
            "replication": rep,
            "scenario": scenario_name,
            "disease": disease_config.name,
            "metrics": metrics,
            "comparative": result.get("comparative", {}),
            "engine_telemetry": result.get("engine_telemetry", {}),
        }
        with open(rep_file, "w") as f:
            json.dump(save_data, f, indent=2, default=str)

        traj_file = os.path.join(rep_dir, f"trajectory_{rep:03d}.csv")
        _save_trajectory_csv(result.get("trajectory"), traj_file)

    return rep, "done", metrics_per_scenario


def run_replications_parallel(
    disease_config, sim_config, output_dir: str, n_replications: int,
    n_workers: int, verbose: bool = True,
):
    """Run all replications across `n_workers` processes.

    Returns a dict {scenario_name: [metrics_list_in_rep_order]} mirroring the
    serial runner's output.
    """
    scenarios = build_scenarios(disease_config, sim_config)
    scenario_names = list(scenarios.keys())

    pending = []
    for rep in range(n_replications):
        for sc in scenario_names:
            if not os.path.exists(
                os.path.join(output_dir, sc, f"rep_{rep:03d}.json")
            ):
                pending.append(rep)
                break

    if verbose:
        print(f"Parallel runner: {n_workers} workers, "
              f"{len(pending)} of {n_replications} replications to compute")

    args_list = [
        (rep, disease_config, sim_config, output_dir, scenario_names)
        for rep in range(n_replications)
    ]

    all_metrics = {sc: [None] * n_replications for sc in scenario_names}

    if n_workers <= 1:
        for args in tqdm(args_list, disable=not verbose):
            rep, status, metrics = _worker_run_one(args)
            for sc, m in metrics.items():
                all_metrics[sc][rep] = m
    else:
        with Pool(processes=n_workers) as pool:
            for rep, status, metrics in tqdm(
                pool.imap_unordered(_worker_run_one, args_list),
                total=len(args_list),
                disable=not verbose,
                desc=f"reps ({n_workers}p)",
            ):
                for sc, m in metrics.items():
                    all_metrics[sc][rep] = m

    # Reload metrics from disk for already-completed reps (skipped by worker)
    for rep in range(n_replications):
        for sc in scenario_names:
            if all_metrics[sc][rep] is not None:
                continue
            rep_file = os.path.join(output_dir, sc, f"rep_{rep:03d}.json")
            if not os.path.exists(rep_file):
                continue
            try:
                with open(rep_file) as f:
                    rec = json.load(f)
                all_metrics[sc][rep] = rec.get("metrics", {})
            except (json.JSONDecodeError, OSError):
                pass

    return {sc: [m for m in lst if m is not None]
            for sc, lst in all_metrics.items()}
