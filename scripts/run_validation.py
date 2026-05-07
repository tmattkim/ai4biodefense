"""Validation suite runner.

Executes the model verification and validation suite across multiple tiers:

    Core (always run):
        L1 — ODE analytical verification (mass conservation, R_eff, final size)
        L2 — ABM-ODE agreement under realism (PI coverage, normalized MAE)
        L3 — Hybrid switching TOST equivalence (peak + attack rate)
        L4 — Per-disease face validity (Kermack-McKendrick final-size eq.)

    Tier 1 (--tier 1, default; defensible-against-critics):
        L4'  — Mode-log telemetry across saved replications
        L5  — Realism feature ablation suite (7 conditions)
        L7  — Counterfactual identification (population/network identity)
        L8  — Effective R0 measurement (next-generation matrix)

    Tier 2 (--tier 2):
        L2b — Stripped-ABM convergence to ODE
        L3b — Hybrid switching invariants (mass / monotonicity / smoothness)
        L6  — Intervention dose-response (Patient + System AI sweeps)

    Tier 3 (--tier 3):
        L10 — Replication adequacy / CI plateau

Usage:
    python scripts/run_validation.py                            # Tier 1 (default)
    python scripts/run_validation.py --tier 2                   # + L2b/L3b/L6
    python scripts/run_validation.py --tier 3 --workers 8       # all layers, 8 procs
    python scripts/run_validation.py --tier 0                   # core only
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import pandas as pd

from config.diseases.covid19 import COVID19Config
from analysis.validation import (
    validate_ode_analytical,
    validate_abm_ode_convergence,
    validate_hybrid_switching,
    validate_face_validity,
)
from analysis.validation_extra import (
    L4_aggregate_mode_log,
    L5_realism_ablations,
    L6_intervention_dose_response,
    L7_counterfactual_identity,
    L8_effective_R0_per_disease,
    L2b_stripped_abm_convergence,
    L3b_hybrid_invariants,
    L10_replication_adequacy,
)


def _save_ode_data(ode_result, output_dir):
    """Save ODE trajectory and summary."""
    result = ode_result['result']
    df = pd.DataFrame({
        'day': result['t'],
        'S': result['S_total'],
        'E': result['E_total'],
        'I': result['I_total'],
        'R': result['R_total'],
    })
    df.to_csv(os.path.join(output_dir, "layer1_ode_trajectory.csv"), index=False)
    summary = {k: v for k, v in ode_result.items() if k != 'result'}
    with open(os.path.join(output_dir, "layer1_ode_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def _save_convergence_data(convergence, output_dir):
    """Save ABM-ODE convergence data (mean + 95% PI band per timestep)."""
    summary = {k: v for k, v in convergence.items()
               if k not in ('ode_result', 'abm_trajectories')}
    with open(os.path.join(output_dir, "layer2_convergence_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    trajs = convergence['abm_trajectories']
    max_len = max(len(t['I']) for t in trajs)
    abm_matrix = np.full((len(trajs), max_len), np.nan)
    for i, t in enumerate(trajs):
        abm_matrix[i, :len(t['I'])] = t['I']

    df = pd.DataFrame({
        'day': np.arange(max_len),
        'abm_mean_I': np.nanmean(abm_matrix, axis=0),
        'abm_p2.5_I': np.nanpercentile(abm_matrix, 2.5, axis=0),
        'abm_p97.5_I': np.nanpercentile(abm_matrix, 97.5, axis=0),
        'abm_std_I': np.nanstd(abm_matrix, axis=0),
    })
    ode_I = convergence['ode_result']['I_total'][:max_len]
    df['ode_I'] = np.nan
    df.loc[:len(ode_I) - 1, 'ode_I'] = ode_I
    df.to_csv(os.path.join(output_dir, "layer2_convergence_curves.csv"), index=False)

    final_R = [t['R'][-1] for t in trajs]
    N = trajs[0]['S'][0] + trajs[0]['E'][0] + trajs[0]['I'][0] + trajs[0]['R'][0]
    ar_df = pd.DataFrame({
        'run': range(len(trajs)),
        'final_recovered': final_R,
        'attack_rate': [r / N for r in final_R],
    })
    ar_df.to_csv(os.path.join(output_dir, "layer2_abm_attack_rates.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Run the simulation framework validation suite",
    )
    parser.add_argument("-n", "--runs", type=int, default=50,
                        help="Number of validation runs (default: 50)")
    parser.add_argument("--tier", type=int, default=1, choices=[0, 1, 2, 3],
                        help="Validation depth: 0=core only (Layers 1-4), "
                             "1=add Tier 1 (L4'/L5/L7/L8), "
                             "2=add Tier 2 (L2b/L3b/L6), "
                             "3=add Tier 3 (L10)")
    parser.add_argument("-w", "--workers", type=int, default=0,
                        help="Number of parallel processes for compute-heavy "
                             "layers (L5/L6/L2b/L3b/L10). 0 = auto.")
    args = parser.parse_args()

    output_dir = "output/validation"
    os.makedirs(output_dir, exist_ok=True)

    disease = COVID19Config()
    n_validation_runs = args.runs

    print("=" * 60)
    print("FRAMEWORK VALIDATION")
    print(f"Disease: {disease.display_name}")
    print(f"Runs per layer: {n_validation_runs}")
    print(f"Tier: {args.tier}")
    print(f"Output: {output_dir}/")
    print("=" * 60)

    # ---- Layer 1: ODE Analytical ----
    print(f"\n[L1] ODE Analytical Verification...")
    ode_result = validate_ode_analytical(disease, n_days=180)
    print(f"  Mass conservation: {'PASS' if ode_result['mass_conservation_valid'] else 'FAIL'}")
    print(f"  Max deviation: {ode_result['max_mass_deviation']:.2e}")
    print(f"  Peak: {ode_result['peak_infections']:.0f} at day {ode_result['peak_day']:.1f}")
    print(f"  Final attack rate: {ode_result['final_attack_rate']:.2%}")
    print(f"  R_eff initial: {ode_result['R_eff_initial']:.2f}")
    _save_ode_data(ode_result, output_dir)

    # ---- Layer 2: ABM-ODE Agreement ----
    print(f"\n[L2] ABM-ODE Agreement under Realism ({n_validation_runs} runs)...")
    convergence = validate_abm_ode_convergence(
        disease, n_runs=n_validation_runs, n_days=180,
    )
    print(f"  Agreement: {'PASS' if convergence['agreement_pass'] else 'FAIL'}")
    print(f"    Normalized MAE: {convergence['normalized_mae']:.3f} "
          f"(threshold {convergence['normalized_mae_threshold']}) "
          f"[{'PASS' if convergence['mae_pass'] else 'FAIL'}]")
    print(f"    PI coverage: {convergence['prediction_interval_coverage']:.2f} "
          f"(threshold {convergence['coverage_threshold']}) "
          f"[{'PASS' if convergence['coverage_pass'] else 'FAIL'}]")
    print(f"    AR gap: {convergence['attack_rate_gap']:.3f} "
          f"[{'PASS' if convergence['attack_rate_pass'] else 'FAIL'}]")
    _save_convergence_data(convergence, output_dir)

    # ---- Layer 3: Hybrid Switching TOST ----
    print(f"\n[L3] Hybrid Switching TOST Equivalence ({n_validation_runs} runs)...")
    hybrid_result = validate_hybrid_switching(
        disease, n_runs=n_validation_runs, n_days=180,
    )
    print(f"  Peak equivalence: {'PASS' if hybrid_result['peak_equivalent'] else 'FAIL'} "
          f"(TOST p={hybrid_result['peak_tost_p_value']:.4f})")
    print(f"  Attack rate equivalence: {'PASS' if hybrid_result['attack_rate_equivalent'] else 'FAIL'} "
          f"(TOST p={hybrid_result['attack_rate_tost_p_value']:.4f})")
    print(f"  Switch fraction: {hybrid_result['switch_fraction']:.2f} "
          f"(mean days in ODE: {hybrid_result['mean_days_in_ode']:.1f})")
    with open(os.path.join(output_dir, "layer3_hybrid_summary.json"), "w") as f:
        json.dump(hybrid_result, f, indent=2)

    # ---- Layer 4: Per-Disease Face Validity ----
    print(f"\n[L4] Per-Disease Face Validity (final-size equation)...")
    face = validate_face_validity()
    print(f"  {'Disease':<10} {'R0':>6} {'AR sim':>8} {'AR analytic':>12} {'gap':>8}")
    for name, d in face['per_disease'].items():
        marker = "PASS" if d['attack_rate_pass'] else ("INFO" if not d['strict_final_size_check'] else "FAIL")
        print(f"  {name:<10} {d['R0']:>6.2f} {d['simulated_attack_rate']:>8.3f} "
              f"{d['analytic_attack_rate']:>12.3f} {d['attack_rate_gap']:>8.3f}  [{marker}]")
    print(f"  Mass conservation across all diseases: "
          f"{'PASS' if face['all_mass_conservation_pass'] else 'FAIL'}")
    with open(os.path.join(output_dir, "layer4_face_validity.json"), "w") as f:
        json.dump(face, f, indent=2)

    # ============================================================
    # Tier 1, 2, 3 layers
    # ============================================================
    extra_pass = {}

    if args.tier >= 1:
        print("\n" + "=" * 60)
        print("TIER 1 — Defensible-against-critics")
        print("=" * 60)

        print("\n[L4-prime] Mode-log telemetry across production replications...")
        L4 = L4_aggregate_mode_log("output")
        with open(os.path.join(output_dir, "L4_mode_log_aggregate.json"), "w") as f:
            json.dump(L4, f, indent=2)
        print(f"  Overall ABM-time fraction (intervention scenarios): "
              f"{L4['overall_intervention_abm_fraction_mean']:.2%}")
        print(f"  L4-prime PASS: {L4['pass']}")
        if L4["n_disease_scenario_pairs"] == 0:
            print("  NOTE: no engine_telemetry data found yet. Run experiments first.")
        extra_pass["L4_mode_log"] = L4["pass"]

        print("\n[L7] Counterfactual identification checks...")
        L7 = L7_counterfactual_identity()
        with open(os.path.join(output_dir, "L7_counterfactual_identity.json"), "w") as f:
            json.dump(L7, f, indent=2)
        print(f"  Population identity: "
              f"{'PASS' if L7['population_identical_across_seed'] else 'FAIL'}")
        print(f"  Network identity:    "
              f"{'PASS' if L7['network_identical_across_seed'] else 'FAIL'}")
        print(f"  Intervention reset:  "
              f"{'PASS' if L7['intervention_reset_correct'] else 'FAIL'}")
        print(f"  Deterministic replay:"
              f" {'PASS' if L7['deterministic_replay_correct'] else 'FAIL'}")
        print(f"  L7 PASS: {L7['pass']}")
        extra_pass["L7"] = L7["pass"]

        print("\n[L8] Effective R0 (NGM) per disease...")
        L8 = L8_effective_R0_per_disease()
        with open(os.path.join(output_dir, "L8_effective_R0.json"), "w") as f:
            json.dump(L8, f, indent=2)
        print(f"  {'Disease':<10} {'nominal':>8} {'measured':>10} {'gap':>7}")
        for name, d in L8['per_disease'].items():
            meas = d['R0_effective_NGM']
            meas_s = f"{meas:.2f}" if meas is not None else "  --"
            gap = d.get('gap_relative')
            gap_s = f"{gap:.2%}" if gap is not None else "  --"
            mk = "OK" if d['close'] else "FLAG"
            print(f"  {name:<10} {d['nominal_R0']:>8.2f} {meas_s:>10} {gap_s:>7}  {mk}")
        print(f"  L8 PASS: {L8['pass']}")
        extra_pass["L8"] = L8["pass"]

        print(f"\n[L5] Realism ablation suite (workers={args.workers or 'auto'})...")
        L5 = L5_realism_ablations(
            n_runs=max(20, min(args.runs, 30)),
            n_workers=args.workers or None,
        )
        with open(os.path.join(output_dir, "L5_realism_ablations.json"), "w") as f:
            json.dump(L5, f, indent=2)
        print(f"  {'Ablation':<28} {'Δpeak':>8} {'ΔAR':>8} {'p_peak':>8}  detected")
        for label, d in L5['per_ablation'].items():
            sig = "YES" if d['feature_has_detectable_effect'] else "no"
            print(f"  {label:<28} {d['rel_peak_shift']:>+7.1%} "
                  f"{d['abs_attack_rate_shift']:>+7.1%} {d['p_peak']:>8.4f}  {sig}")
        print(f"  L5 PASS: {L5['pass']}  "
              f"({L5['n_ablations_with_detectable_effect']}/"
              f"{L5['n_ablations_total']} features have detectable effect)")
        extra_pass["L5"] = L5["pass"]

    if args.tier >= 2:
        print("\n" + "=" * 60)
        print("TIER 2 — Engine math + intervention dose-response")
        print("=" * 60)

        print(f"\n[L2b] Stripped-ABM convergence to ODE ({args.runs} runs, "
              f"workers={args.workers or 'auto'})...")
        L2b = L2b_stripped_abm_convergence(
            n_runs=args.runs, n_workers=args.workers or None,
        )
        with open(os.path.join(output_dir, "L2b_stripped_abm.json"), "w") as f:
            json.dump(L2b, f, indent=2)
        print(f"  Stripped ABM peak={L2b['stripped_abm_mean_peak']:.0f} "
              f"vs ODE peak={L2b['ode_peak']:.0f}")
        print(f"  Normalized MAE: {L2b['normalized_mae']:.3f}, "
              f"PI coverage: {L2b['prediction_interval_coverage']:.2%}")
        print(f"  L2b PASS: {L2b['pass']}")
        extra_pass["L2b"] = L2b["pass"]

        print(f"\n[L3b] Hybrid switching invariants ({max(20, args.runs//2)} runs, "
              f"workers={args.workers or 'auto'})...")
        L3b = L3b_hybrid_invariants(
            n_runs=max(20, args.runs // 2), n_workers=args.workers or None,
        )
        with open(os.path.join(output_dir, "L3b_hybrid_invariants.json"), "w") as f:
            json.dump(L3b, f, indent=2)
        print(f"  Mass violations:       {L3b['mass_violations']}/{L3b['n_runs']}")
        print(f"  Monotone-R violations: {L3b['monotone_R_violations']}/{L3b['n_runs']}")
        print(f"  Smoothness violations: {L3b['smoothness_violations']}/{L3b['n_runs']}")
        print(f"  L3b PASS: {L3b['pass']}")
        extra_pass["L3b"] = L3b["pass"]

        print(f"\n[L6] Intervention dose-response sweeps "
              f"({max(20, args.runs//2)} runs/point, "
              f"workers={args.workers or 'auto'})...")
        L6 = L6_intervention_dose_response(
            n_runs=max(20, args.runs // 2), n_workers=args.workers or None,
        )
        with open(os.path.join(output_dir, "L6_dose_response.json"), "w") as f:
            json.dump(L6, f, indent=2)
        for label, d in L6['by_dial'].items():
            mark = "OK" if d['monotonic_in_expected_direction'] else "FAIL"
            rho = d['rho_peak']
            rho_s = f"ρ={rho:.2f}" if rho is not None else "ρ=--"
            print(f"  {label:<35} {rho_s}  {mark}")
        print(f"  L6 PASS: {L6['pass']}")
        extra_pass["L6"] = L6["pass"]

    if args.tier >= 3:
        print("\n" + "=" * 60)
        print("TIER 3 — Replication adequacy")
        print("=" * 60)

        print(f"\n[L10] Replication adequacy / CI plateau "
              f"(workers={args.workers or 'auto'})...")
        L10 = L10_replication_adequacy(n_workers=args.workers or None)
        with open(os.path.join(output_dir, "L10_replication_adequacy.json"), "w") as f:
            json.dump(L10, f, indent=2)
        print(f"  {'n_reps':>8} {'peak_mean':>10} {'peak_CI_w':>10} "
              f"{'AR_mean':>8} {'AR_CI_w':>9}")
        for entry in L10['by_n_reps']:
            print(f"  {entry['n']:>8} {entry['peak_mean']:>10.0f} "
                  f"{entry['peak_ci_width']:>10.0f} "
                  f"{entry['attack_rate_mean']:>8.3f} "
                  f"{entry['attack_rate_ci_width']:>9.3f}")
        print(f"  L10 PASS: {L10['pass']}")
        extra_pass["L10"] = L10["pass"]

    # ---- Overall summary ----
    all_pass = (
        ode_result['mass_conservation_valid'] and
        convergence['agreement_pass'] and
        hybrid_result['peak_equivalent'] and
        hybrid_result['attack_rate_equivalent'] and
        face['face_validity_pass'] and
        all(extra_pass.values())
    )
    print("\n" + "=" * 60)
    status = "ALL PASS" if all_pass else "SOME FAILURES (see per-layer interpretation)"
    print(f"VALIDATION RESULT: {status}")
    print("=" * 60)

    print(f"\nOutput files in {output_dir}/:")
    for f in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, f))
        size_str = (f"{size/1024/1024:.1f} MB" if size > 1024*1024
                    else f"{size/1024:.1f} KB" if size > 1024
                    else f"{size} B")
        print(f"  {f:<45} {size_str:>10}")

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
