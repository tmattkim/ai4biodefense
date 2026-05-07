"""Run experiment for a single disease. Supports all 5 diseases.

Usage:
    python scripts/run_disease.py covid19 -n 100
    python scripts/run_disease.py influenza -n 10
    python scripts/run_disease.py ebola -n 50
    python scripts/run_disease.py dengue -n 100
    python scripts/run_disease.py measles -n 100
    python scripts/run_disease.py all -n 100

Use --workers N for multi-process parallelism (recommended on multi-core
machines): the runner is resumability-safe, so interruptions are recoverable.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from config.base import SimulationConfig
from experiments.runner import ExperimentRunner


def _load_config(name):
    """Lazy-load disease configs to avoid import errors for missing ones."""
    if name == "covid19":
        from config.diseases.covid19 import COVID19Config
        return COVID19Config()
    elif name == "influenza":
        from config.diseases.influenza import InfluenzaConfig
        return InfluenzaConfig()
    elif name == "ebola":
        from config.diseases.ebola import EbolaConfig
        return EbolaConfig()
    elif name == "dengue":
        from config.diseases.dengue import DengueConfig
        return DengueConfig()
    elif name == "measles":
        from config.diseases.measles import MeaslesConfig
        return MeaslesConfig()
    else:
        raise ValueError(f"Unknown disease: {name}")


ALL_DISEASES = ["covid19", "influenza", "ebola", "dengue", "measles"]


def run_single_disease(disease_name, n_replications, output_base, quiet, n_workers=1):
    """Run all 4 intervention scenarios for one disease."""
    disease = _load_config(disease_name)
    sim_config = SimulationConfig(num_replications=n_replications)
    output_dir = os.path.join(output_base, disease_name)

    print(f"\n{'='*60}")
    print(f"  {disease.display_name}")
    print(f"  R0={disease.R0}  Horizon={sim_config.get_time_horizon(disease.name)}d")
    print(f"  Replications: {n_replications}  Workers: {n_workers}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    runner = ExperimentRunner(disease, sim_config)
    summary = runner.run_all(
        output_dir=output_dir,
        n_replications=n_replications,
        verbose=not quiet,
        n_workers=n_workers,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run intervention experiments for a specific disease",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python scripts/run_disease.py covid19 -n 10\n"
               "  python scripts/run_disease.py influenza ebola -n 50\n"
               "  python scripts/run_disease.py all -n 100\n",
    )
    parser.add_argument(
        "diseases", nargs="+",
        help="Disease(s) to run: covid19, influenza, ebola, dengue, measles, or 'all'",
    )
    parser.add_argument("-n", "--replications", type=int, default=100,
                        help="Number of replications (default: 100)")
    parser.add_argument("-o", "--output", type=str, default="output",
                        help="Base output directory (default: output)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress progress bars")
    parser.add_argument("-w", "--workers", type=int, default=1,
                        help="Number of parallel processes (default: 1). "
                             "Set to ~os.cpu_count() for max speedup. "
                             "Resumability-safe.")
    args = parser.parse_args()

    diseases = []
    for d in args.diseases:
        if d.lower() == "all":
            diseases = ALL_DISEASES
            break
        diseases.append(d.lower())

    for d in diseases:
        if d not in ALL_DISEASES:
            print(f"Error: Unknown disease '{d}'. Choose from: {ALL_DISEASES}")
            sys.exit(1)

    print(f"Running: {diseases}")
    print(f"Replications: {args.replications}")

    summaries = {}
    for disease_name in diseases:
        summaries[disease_name] = run_single_disease(
            disease_name, args.replications, args.output, args.quiet,
            n_workers=args.workers,
        )

    print(f"\nAll results saved to: {args.output}/")
    return summaries


if __name__ == "__main__":
    main()
