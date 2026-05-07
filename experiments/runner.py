"""Experiment orchestration: run replications across scenarios.

Saves per-replication trajectory CSVs, metrics JSON, and summary statistics.
"""

import json
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from config.base import SimulationConfig
from experiments.scenario import build_scenarios
from experiments.counterfactual import CounterfactualAnalyzer
from analysis.statistics import paired_comparison, bootstrap_ci


class ExperimentRunner:
    """Orchestrate multiple replications across intervention scenarios."""

    def __init__(self, disease_config, sim_config=None):
        self.disease_config = disease_config
        self.sim_config = sim_config or SimulationConfig()
        self.scenarios = build_scenarios(disease_config, self.sim_config)

    def _load_completed_replications(self, output_dir, n_reps):
        """Scan output dir for completed rep files and load their metrics."""
        completed = {}
        for scenario in self.scenarios:
            rep_dir = os.path.join(output_dir, scenario)
            if not os.path.isdir(rep_dir):
                continue
            for rep in range(n_reps):
                rep_file = os.path.join(rep_dir, f"rep_{rep:03d}.json")
                if os.path.exists(rep_file):
                    try:
                        with open(rep_file) as f:
                            data = json.load(f)
                        if rep not in completed:
                            completed[rep] = {}
                        completed[rep][scenario] = data.get("metrics", {})
                    except (json.JSONDecodeError, KeyError):
                        pass
        full_reps = {r for r, sc in completed.items()
                     if set(sc.keys()) == set(self.scenarios.keys())}
        return full_reps, completed

    def run_all(self, output_dir: str, n_replications: int = None,
                verbose: bool = True, n_workers: int = 1):
        """Run all scenarios for all replications.

        Args:
            output_dir: Directory to save results.
            n_replications: Override number of replications.
            verbose: Print progress.
            n_workers: If > 1, dispatches replications across that many
                processes. Each worker does its own file I/O. Resumability-safe.

        Returns:
            Consolidated results dict.
        """
        n_reps = n_replications or self.sim_config.num_replications
        os.makedirs(output_dir, exist_ok=True)

        # Multi-process path: defer to experiments.parallel, then re-load
        # metrics from disk and continue with summary.
        if n_workers and n_workers > 1:
            from experiments.parallel import run_replications_parallel
            all_metrics = run_replications_parallel(
                self.disease_config, self.sim_config, output_dir,
                n_reps, n_workers, verbose=verbose,
            )
            summary = self._compute_summary(all_metrics)
            with open(os.path.join(output_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2, default=str)
            self._save_metrics_csv(all_metrics, output_dir)
            if verbose:
                self._print_summary(summary)
                self._print_output_files(output_dir)
            return summary

        # Single-process path
        analyzer = CounterfactualAnalyzer(self.disease_config, self.sim_config)
        all_metrics = {s: [] for s in self.scenarios}

        completed_reps, loaded = self._load_completed_replications(
            output_dir, n_reps
        )
        for rep in sorted(completed_reps):
            for scenario in self.scenarios:
                all_metrics[scenario].append(loaded[rep][scenario])

        remaining = [r for r in range(n_reps) if r not in completed_reps]

        if verbose:
            print(f"Running {n_reps} replications for "
                  f"{self.disease_config.display_name}")
            print(f"Scenarios: {list(self.scenarios.keys())}")
            if completed_reps:
                print(f"Resuming: {len(completed_reps)} replications already "
                      f"complete, {len(remaining)} remaining")

        for rep in tqdm(remaining, desc="Replications", disable=not verbose):
            rep_results = analyzer.run_counterfactual_set(
                rep, self.scenarios, verbose=False,
            )

            for scenario_name, result in rep_results.items():
                metrics = result.get("metrics", {})
                all_metrics[scenario_name].append(metrics)

                rep_dir = os.path.join(output_dir, scenario_name)
                os.makedirs(rep_dir, exist_ok=True)

                rep_file = os.path.join(rep_dir, f"rep_{rep:03d}.json")
                save_data = {
                    "replication": rep,
                    "scenario": scenario_name,
                    "disease": self.disease_config.name,
                    "metrics": metrics,
                    "comparative": result.get("comparative", {}),
                    "engine_telemetry": result.get("engine_telemetry", {}),
                }
                with open(rep_file, "w") as f:
                    json.dump(save_data, f, indent=2, default=str)

                trajectory = result.get("trajectory")
                if trajectory is not None:
                    traj_file = os.path.join(rep_dir, f"trajectory_{rep:03d}.csv")
                    _save_trajectory_csv(trajectory, traj_file)

        summary = self._compute_summary(all_metrics)
        summary_file = os.path.join(output_dir, "summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        self._save_metrics_csv(all_metrics, output_dir)

        if verbose:
            self._print_summary(summary)
            self._print_output_files(output_dir)

        return summary

    def _compute_summary(self, all_metrics: dict) -> dict:
        """Compute paired comparisons and summary statistics."""
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "disease": self.disease_config.name,
            "disease_display": self.disease_config.display_name,
            "R0": self.disease_config.R0,
            "n_replications": len(all_metrics.get("baseline", [])),
            "population_size": self.sim_config.population_size,
            "per_scenario": {},
            "comparisons": {},
        }

        for scenario, metrics_list in all_metrics.items():
            if not metrics_list:
                continue
            ar = np.array([m.get("cumulative_attack_rate", 0) for m in metrics_list])
            peak = np.array([m.get("peak_incidence", 0) for m in metrics_list])
            timing = np.array([m.get("peak_timing", 0) for m in metrics_list])
            hosp = np.array([m.get("hospital_days", 0) for m in metrics_list])
            total_inf = np.array([m.get("total_infections", 0) for m in metrics_list])
            duration = np.array([m.get("epidemic_duration", 0) for m in metrics_list])

            summary["per_scenario"][scenario] = {
                "attack_rate": {
                    "mean": float(ar.mean()), "std": float(ar.std()),
                    "ci": bootstrap_ci(ar),
                },
                "peak_incidence": {
                    "mean": float(peak.mean()), "std": float(peak.std()),
                    "ci": bootstrap_ci(peak),
                },
                "peak_timing": {
                    "mean": float(timing.mean()), "std": float(timing.std()),
                },
                "hospital_days": {
                    "mean": float(hosp.mean()), "std": float(hosp.std()),
                    "ci": bootstrap_ci(hosp),
                },
                "total_infections": {
                    "mean": float(total_inf.mean()), "std": float(total_inf.std()),
                },
                "epidemic_duration": {
                    "mean": float(duration.mean()), "std": float(duration.std()),
                },
            }

        scenario_order = ["baseline", "patient_ai", "system_ai", "combined"]
        available = [s for s in scenario_order if s in all_metrics and all_metrics[s]]

        def _extract(scenario, key):
            return np.array([m.get(key, 0) for m in all_metrics[scenario]])

        for i, s_a in enumerate(available):
            for s_b in available[i + 1:]:
                n = min(len(all_metrics[s_a]), len(all_metrics[s_b]))
                pair_key = f"{s_a}_vs_{s_b}"
                summary["comparisons"][pair_key] = {
                    "attack_rate": paired_comparison(
                        _extract(s_a, "cumulative_attack_rate")[:n],
                        _extract(s_b, "cumulative_attack_rate")[:n],
                    ),
                    "peak_incidence": paired_comparison(
                        _extract(s_a, "peak_incidence")[:n],
                        _extract(s_b, "peak_incidence")[:n],
                    ),
                    "hospital_days": paired_comparison(
                        _extract(s_a, "hospital_days")[:n],
                        _extract(s_b, "hospital_days")[:n],
                    ),
                }

        return summary

    def _save_metrics_csv(self, all_metrics: dict, output_dir: str):
        """Save all replications as a single CSV for analysis."""
        rows = []
        for scenario, metrics_list in all_metrics.items():
            for rep, metrics in enumerate(metrics_list):
                row = {
                    "disease": self.disease_config.name,
                    "scenario": scenario,
                    "replication": rep,
                }
                row.update(metrics)
                row.pop("R_eff_trajectory", None)
                rows.append(row)
        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, "all_replications.csv")
        df.to_csv(csv_path, index=False)

    def _print_summary(self, summary: dict):
        """Print formatted summary to console."""
        print("\n" + "=" * 70)
        print(f"EXPERIMENT SUMMARY: {summary['disease_display']}")
        print(f"Replications: {summary['n_replications']}")
        print("=" * 70)

        for scenario, stats in summary.get("per_scenario", {}).items():
            print(f"\n  {scenario}:")
            ar = stats.get("attack_rate", {})
            peak = stats.get("peak_incidence", {})
            hosp = stats.get("hospital_days", {})
            dur = stats.get("epidemic_duration", {})
            print(f"    Attack rate:  {ar.get('mean', 0):.2%} "
                  f"(±{ar.get('std', 0):.2%})")
            print(f"    Peak:         {peak.get('mean', 0):.0f} "
                  f"(±{peak.get('std', 0):.0f})")
            print(f"    Hospital-days:{hosp.get('mean', 0):.0f} "
                  f"(±{hosp.get('std', 0):.0f})")
            print(f"    Duration:     {dur.get('mean', 0):.0f}d "
                  f"(±{dur.get('std', 0):.0f}d)")

        LABELS = {
            'baseline': 'Baseline', 'patient_ai': 'Patient AI',
            'system_ai': 'System AI', 'combined': 'Combined',
        }
        print("\n  Paired Comparisons:")
        for pair_key, comps in summary.get("comparisons", {}).items():
            parts = pair_key.split("_vs_")
            label_a = LABELS.get(parts[0], parts[0])
            label_b = LABELS.get(parts[1], parts[1]) if len(parts) > 1 else pair_key
            ar_comp = comps.get("attack_rate", {})
            peak_comp = comps.get("peak_incidence", {})
            hosp_comp = comps.get("hospital_days", {})
            sig = lambda p: "*" if p < 0.05 else ""
            print(f"\n    {label_a} vs {label_b}:")
            print(f"      Attack rate:  diff={ar_comp.get('mean_diff', 0):.4f} "
                  f"p={ar_comp.get('p_value', 1):.4f}{sig(ar_comp.get('p_value', 1))} "
                  f"d={ar_comp.get('cohens_d', 0):.2f}")
            print(f"      Peak:         diff={peak_comp.get('mean_diff', 0):.0f} "
                  f"p={peak_comp.get('p_value', 1):.4f}{sig(peak_comp.get('p_value', 1))} "
                  f"d={peak_comp.get('cohens_d', 0):.2f}")
            print(f"      Hospital-days:diff={hosp_comp.get('mean_diff', 0):.0f} "
                  f"p={hosp_comp.get('p_value', 1):.4f}{sig(hosp_comp.get('p_value', 1))} "
                  f"d={hosp_comp.get('cohens_d', 0):.2f}")
        print("\n" + "=" * 70)

    def _print_output_files(self, output_dir):
        """List all output files with sizes."""
        print(f"\nOutput files in {output_dir}/:")
        total_size = 0
        for root, dirs, files in os.walk(output_dir):
            for f in sorted(files):
                path = os.path.join(root, f)
                size = os.path.getsize(path)
                total_size += size
                rel = os.path.relpath(path, output_dir)
                if size > 1024 * 1024:
                    size_str = f"{size / 1024 / 1024:.1f} MB"
                elif size > 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size} B"
                if 'rep_' not in f and 'trajectory_' not in f:
                    print(f"  {rel:<50} {size_str:>10}")
        n_rep_files = sum(1 for r, d, fs in os.walk(output_dir)
                         for f in fs if 'rep_' in f or 'trajectory_' in f)
        if n_rep_files:
            print(f"  + {n_rep_files} per-replication files (metrics + trajectories)")
        if total_size > 1024 * 1024:
            print(f"  Total: {total_size / 1024 / 1024:.1f} MB")


def _save_trajectory_csv(trajectory: dict, path: str):
    """Save a simulation trajectory to CSV."""
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
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
