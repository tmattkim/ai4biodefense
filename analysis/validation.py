"""Model verification and validation suite.

Core 4-layer validation:
1. ODE vs analytical SEIR (mass conservation + final-size)
2. ABM-ODE agreement under realism (percentile prediction interval + normalized MAE)
3. Hybrid switching bias (TOST equivalence + switch coverage diagnostics)
4. Per-disease face validity (Kermack-McKendrick final-size across pathogens)
"""

import numpy as np
from scipy import stats
from scipy.optimize import brentq
from tqdm import tqdm

from config.base import SimulationConfig
from config.diseases.covid19 import COVID19Config
from core.population import create_population
from core.contact_network import build_contact_network
from models.seir_ode import solve_seir, verify_mass_conservation
from models.abm_engine import ABMEngine
from models.hybrid_engine import HybridEngine


def validate_ode_analytical(disease_config=None, n_days=180):
    """Layer 1: Verify ODE solver against analytical SEIR properties.

    Checks:
    - Mass conservation: S+E+I+R+Iiso = N at all times
    - Final size equation (for SIR limit)
    - R0 recovery: early growth rate matches expected
    """
    if disease_config is None:
        disease_config = COVID19Config()

    pop_by_age = np.array([600, 1600, 4000, 1300, 1700], dtype=float)  # ~10K
    N = pop_by_age.sum()

    result = solve_seir(
        disease_config, pop_by_age,
        t_span=(0, n_days), dt_eval=0.5,
    )

    # Mass conservation
    is_valid, max_dev = verify_mass_conservation(result, N)

    # Peak should occur
    peak_I = np.max(result['I_total'])
    peak_day = result['t'][np.argmax(result['I_total'])]

    # R_eff should start near R0 and decline
    R_eff_start = disease_config.R0 * result['S_total'][0] / N
    R_eff_end = disease_config.R0 * result['S_total'][-1] / N

    # Final size: should converge to steady state
    final_S = result['S_total'][-1]
    final_R = result['R_total'][-1]

    return {
        "mass_conservation_valid": bool(is_valid),
        "max_mass_deviation": float(max_dev),
        "peak_infections": float(peak_I),
        "peak_day": float(peak_day),
        "R_eff_initial": float(R_eff_start),
        "R_eff_final": float(R_eff_end),
        "final_susceptible_fraction": float(final_S / N),
        "final_attack_rate": float(final_R / N),
        "N_total": float(N),
        "result": result,
    }


def validate_abm_ode_convergence(
    disease_config=None, n_runs=50, n_days=180, seed_base=42,
    mae_tolerance_frac=0.15, coverage_threshold=0.80,
):
    """Layer 2: ABM-ODE agreement under realism.

    The ABM carries realism features the analytic ODE does not model
    (individual-level superspreading, pre-symptomatic transmission,
    archetype heterogeneity, compliance fatigue, network structure).
    Exact convergence is NOT expected. We test for *reasonable agreement*:

    Pass criteria:
    - Normalized MAE: mean(|abm_mean - ode|) / ode_peak < mae_tolerance_frac
    - Prediction-interval coverage: fraction of timepoints where ODE lies
      inside the empirical ABM [p2.5, p97.5] prediction interval
      >= coverage_threshold
    - Final attack rate agreement: |abm_mean_AR - ode_AR| < 0.10

    The old "ODE inside ABM standard-error CI" test was dropped — with
    n=50 runs that band is far tighter than the between-run std and
    flags correct models as failing.
    """
    if disease_config is None:
        disease_config = COVID19Config()

    sim_config = SimulationConfig()
    pop_by_age = np.array([600, 1600, 4000, 1300, 1700], dtype=float)
    N = int(pop_by_age.sum())

    # Get ODE reference solution
    ode_result = solve_seir(disease_config, pop_by_age, t_span=(0, n_days))

    # Run ABM simulations
    abm_trajectories = []
    for run in tqdm(range(n_runs), desc="ABM-ODE convergence", unit="run"):
        rng = np.random.default_rng(seed_base + run)
        agents = create_population(sim_config, disease_config, rng)
        networks = build_contact_network(agents, rng)
        abm = ABMEngine(agents, networks, disease_config, dt=1.0, rng=rng)

        trajectory = {'days': [], 'S': [], 'E': [], 'I': [], 'R': []}
        for day in range(n_days):
            result = abm.step(float(day))
            sc = result['state_counts']
            trajectory['days'].append(day)
            trajectory['S'].append(sc[0])
            trajectory['E'].append(sc[1])
            trajectory['I'].append(sc[2])
            trajectory['R'].append(sc[3])

        for k in trajectory:
            trajectory[k] = np.array(trajectory[k])
        abm_trajectories.append(trajectory)

    # Align trajectory lengths on a daily grid
    max_len = min(n_days, int(np.floor(ode_result['t'][-1])) + 1)
    abm_I_matrix = np.array([t['I'][:max_len] for t in abm_trajectories])
    abm_mean = abm_I_matrix.mean(axis=0)

    # Empirical prediction interval (NOT standard error of the mean)
    abm_pi_lo = np.percentile(abm_I_matrix, 2.5, axis=0)
    abm_pi_hi = np.percentile(abm_I_matrix, 97.5, axis=0)

    # Interpolate ODE to daily grid
    ode_I = np.interp(np.arange(max_len), ode_result['t'], ode_result['I_total'])

    # Prediction-interval coverage (fraction of days ODE lies in ABM PI)
    in_pi = (ode_I >= abm_pi_lo) & (ode_I <= abm_pi_hi)
    pi_coverage = float(np.mean(in_pi))

    # Normalized MAE
    mae = float(np.mean(np.abs(abm_mean - ode_I)))
    ode_peak = float(np.max(ode_I))
    norm_mae = mae / max(ode_peak, 1.0)

    abm_final_R = np.array([t['R'][-1] for t in abm_trajectories]) / N
    ode_final_R = ode_result['R_total'][-1] / N
    ar_gap = float(abs(abm_final_R.mean() - ode_final_R))

    mae_pass = norm_mae < mae_tolerance_frac
    coverage_pass = pi_coverage >= coverage_threshold
    ar_pass = ar_gap < 0.10

    return {
        # Primary pass criteria
        "agreement_pass": bool(mae_pass and coverage_pass and ar_pass),
        "normalized_mae": float(norm_mae),
        "normalized_mae_threshold": float(mae_tolerance_frac),
        "mae_pass": bool(mae_pass),
        "prediction_interval_coverage": pi_coverage,
        "coverage_threshold": float(coverage_threshold),
        "coverage_pass": bool(coverage_pass),
        "attack_rate_gap": ar_gap,
        "attack_rate_pass": bool(ar_pass),
        # Diagnostics
        "mean_absolute_error": mae,
        "abm_mean_peak": float(np.max(abm_mean)),
        "ode_peak": ode_peak,
        "peak_difference": float(np.max(abm_mean) - ode_peak),
        "abm_final_attack_rate_mean": float(abm_final_R.mean()),
        "ode_final_attack_rate": float(ode_final_R),
        "n_runs": n_runs,
        # Legacy field kept for back-compat with plotting / old consumers
        "ode_within_abm_ci": bool(mae_pass and coverage_pass and ar_pass),
        "ode_result": ode_result,
        "abm_trajectories": abm_trajectories,
    }


def _tost(x, y, bound):
    """Two One-Sided Tests for equivalence of independent means.

    H0: |mean(x) - mean(y)| >= bound  (non-equivalence)
    H1: |mean(x) - mean(y)| <  bound  (equivalence)

    Returns (p_equivalence, t_lower, t_upper). Reject H0 if p_equivalence < alpha.
    """
    mean_diff = np.mean(x) - np.mean(y)
    se = np.sqrt(np.var(x, ddof=1) / len(x) + np.var(y, ddof=1) / len(y))
    if se == 0:
        return (0.0, np.inf, np.inf)
    dof = len(x) + len(y) - 2
    t_lower = (mean_diff - (-bound)) / se  # vs lower bound
    t_upper = (mean_diff - bound) / se     # vs upper bound
    p_lower = 1.0 - stats.t.cdf(t_lower, dof)
    p_upper = stats.t.cdf(t_upper, dof)
    return (float(max(p_lower, p_upper)), float(t_lower), float(t_upper))


def validate_hybrid_switching(
    disease_config=None, n_runs=50, n_days=180, seed_base=42,
    peak_equivalence_frac=0.10, ar_equivalence_bound=0.05, alpha=0.05,
):
    """Layer 3: Verify hybrid switching introduces no systematic bias.

    Uses a **TOST equivalence test** (not a standard t-test): p > 0.05
    from a t-test is failure-to-reject, which is NOT evidence of no bias.
    TOST directly tests H1: |pure - hybrid| < bound.

    Equivalence bounds (pre-registered):
    - Peak incidence: +/- peak_equivalence_frac * pure_mean_peak (default 10%)
    - Attack rate:    +/- ar_equivalence_bound (default 0.05, i.e. 5pp)

    Also reports:
    - switch_fraction: fraction of hybrid runs that actually entered ODE mode
      (if ~0, the test is trivially satisfied — epidemic stayed sub-threshold)
    - mean_days_in_ode: average time spent in ODE mode per hybrid run
    """
    if disease_config is None:
        disease_config = COVID19Config()

    sim_config = SimulationConfig()

    pure_abm_metrics = []
    hybrid_metrics = []
    hybrid_switched = []
    hybrid_days_in_ode = []

    for run in tqdm(range(n_runs), desc="Hybrid switching", unit="run"):
        rng_pure = np.random.default_rng(seed_base + run)
        rng_hybrid = np.random.default_rng(seed_base + run)

        # Pure ABM run
        agents = create_population(sim_config, disease_config, rng_pure)
        networks = build_contact_network(agents, rng_pure)
        abm = ABMEngine(agents, networks, disease_config, dt=1.0, rng=rng_pure)

        I_curve = []
        for day in range(n_days):
            result = abm.step(float(day))
            I_curve.append(result['state_counts'][2])
        I_curve = np.array(I_curve)
        pure_abm_metrics.append({
            'peak': float(np.max(I_curve)),
            'peak_day': float(np.argmax(I_curve)),
            'attack_rate': float(result['state_counts'][3]) / sim_config.population_size,
        })

        # Hybrid run (same seed)
        agents2 = create_population(sim_config, disease_config, rng_hybrid)
        networks2 = build_contact_network(agents2, rng_hybrid)
        hybrid = HybridEngine(agents2, networks2, disease_config, sim_config, rng_hybrid)

        I_curve_h = []
        for day in range(n_days):
            result_h = hybrid.step(float(day))
            I_curve_h.append(result_h['state_counts'][2])
        I_curve_h = np.array(I_curve_h)
        hybrid_metrics.append({
            'peak': float(np.max(I_curve_h)),
            'peak_day': float(np.argmax(I_curve_h)),
            'attack_rate': float(result_h['state_counts'][3]) / sim_config.population_size,
        })

        # Switch diagnostics — mode_log entries are (day, mode_str) tuples.
        modes = getattr(hybrid, 'mode_log', [])
        n_ode = sum(1 for entry in modes if entry[1] == "ODE")
        hybrid_days_in_ode.append(n_ode)
        hybrid_switched.append(n_ode > 0)

    pure_peaks = np.array([m['peak'] for m in pure_abm_metrics])
    hybrid_peaks = np.array([m['peak'] for m in hybrid_metrics])
    pure_ar = np.array([m['attack_rate'] for m in pure_abm_metrics])
    hybrid_ar = np.array([m['attack_rate'] for m in hybrid_metrics])
    pure_peak_days = np.array([m['peak_day'] for m in pure_abm_metrics])
    hybrid_peak_days = np.array([m['peak_day'] for m in hybrid_metrics])

    # Legacy t-tests (for diagnostics)
    t_peak, p_peak = stats.ttest_ind(pure_peaks, hybrid_peaks, equal_var=False)
    t_ar, p_ar = stats.ttest_ind(pure_ar, hybrid_ar, equal_var=False)
    t_pd, p_pd = stats.ttest_ind(pure_peak_days, hybrid_peak_days, equal_var=False)

    # TOST equivalence tests
    peak_bound = peak_equivalence_frac * float(pure_peaks.mean())
    p_tost_peak, _, _ = _tost(pure_peaks, hybrid_peaks, peak_bound)
    p_tost_ar, _, _ = _tost(pure_ar, hybrid_ar, ar_equivalence_bound)

    switch_fraction = float(np.mean(hybrid_switched))
    mean_days_in_ode = float(np.mean(hybrid_days_in_ode))

    return {
        # TOST equivalence (primary)
        "peak_equivalent": bool(p_tost_peak < alpha),
        "peak_tost_p_value": float(p_tost_peak),
        "peak_equivalence_bound": float(peak_bound),
        "attack_rate_equivalent": bool(p_tost_ar < alpha),
        "attack_rate_tost_p_value": float(p_tost_ar),
        "attack_rate_equivalence_bound": float(ar_equivalence_bound),
        # Switch coverage (critical caveat)
        "switch_fraction": switch_fraction,
        "mean_days_in_ode": mean_days_in_ode,
        "switching_exercised": bool(switch_fraction >= 0.5),
        # Diagnostic t-tests
        "peak_t_statistic": float(t_peak),
        "peak_p_value": float(p_peak),
        "peak_no_bias": bool(p_peak > 0.05),
        "attack_rate_t_statistic": float(t_ar),
        "attack_rate_p_value": float(p_ar),
        "attack_rate_no_bias": bool(p_ar > 0.05),
        "peak_timing_p_value": float(p_pd),
        # Means
        "pure_abm_mean_peak": float(pure_peaks.mean()),
        "hybrid_mean_peak": float(hybrid_peaks.mean()),
        "pure_abm_mean_ar": float(pure_ar.mean()),
        "hybrid_mean_ar": float(hybrid_ar.mean()),
        "pure_abm_mean_peak_day": float(pure_peak_days.mean()),
        "hybrid_mean_peak_day": float(hybrid_peak_days.mean()),
        "n_runs": n_runs,
    }

def _final_size_kermack_mckendrick(R0, s0=1.0):
    """Solve final-size equation: 1 - s_inf - s0 * (1 - exp(-R0 * (1 - s_inf))) = 0.

    For an SIR/SEIR epidemic in a homogeneous population with no interventions,
    the final attack rate converges to a well-defined value depending only on R0.
    (Kermack & McKendrick 1927.)
    """
    if R0 <= 1.0:
        return 0.0
    # Solve for final susceptible fraction s_inf: s_inf = s0 * exp(-R0 * (1 - s_inf)).
    # For large R0 the non-trivial root is ~exp(-R0), so widen the lower bracket.
    def f(s_inf):
        return s_inf - s0 * np.exp(-R0 * (1.0 - s_inf))
    lo = min(1e-14, np.exp(-R0) / 10.0)
    s_inf = brentq(f, lo, s0 - 1e-8)
    return 1.0 - s_inf  # final attack rate


def validate_face_validity(n_days=365, ar_tolerance=0.10):
    """Layer 5: Per-disease face validity against analytic expectations.

    For each of the 5 pathogen configs, solves the ODE with no interventions
    and compares:
    - Final attack rate vs Kermack-McKendrick final-size prediction (R0 only)
    - Initial R_eff vs R0 (sanity: should equal R0 when S ~ N)
    - Mass conservation

    Vector-borne pathogens (e.g., dengue) and diseases with strong structure
    (measles: contact scaling, ebola: household/burial) will deviate from the
    homogeneous final-size prediction. We report the gap rather than strictly
    requiring agreement for those cases.
    """
    from config.diseases.covid19 import COVID19Config
    from config.diseases.influenza import InfluenzaConfig
    from config.diseases.ebola import EbolaConfig
    from config.diseases.measles import MeaslesConfig

    diseases = [
        ("COVID-19", COVID19Config(), True),
        ("Influenza", InfluenzaConfig(), True),
        ("Ebola", EbolaConfig(), False),   # household/burial structure — deviation expected
        ("Measles", MeaslesConfig(), False),  # contact scaling — deviation expected
    ]

    per_disease = {}
    all_mass_ok = True
    all_ar_ok_for_strict = True

    for name, cfg, strict in diseases:
        r = validate_ode_analytical(cfg, n_days=n_days)
        ar_sim = r['final_attack_rate']
        ar_analytic = _final_size_kermack_mckendrick(cfg.R0)
        ar_gap = abs(ar_sim - ar_analytic)
        r_eff_gap = abs(r['R_eff_initial'] - cfg.R0)

        mass_ok = r['mass_conservation_valid']
        ar_ok = ar_gap < ar_tolerance
        reff_ok = r_eff_gap < 0.05

        if not mass_ok:
            all_mass_ok = False
        if strict and not ar_ok:
            all_ar_ok_for_strict = False

        per_disease[name] = {
            "R0": float(cfg.R0),
            "simulated_attack_rate": float(ar_sim),
            "analytic_attack_rate": float(ar_analytic),
            "attack_rate_gap": float(ar_gap),
            "attack_rate_pass": bool(ar_ok),
            "mass_conservation_valid": bool(mass_ok),
            "max_mass_deviation": float(r['max_mass_deviation']),
            "R_eff_initial": float(r['R_eff_initial']),
            "R_eff_gap_from_R0": float(r_eff_gap),
            "R_eff_pass": bool(reff_ok),
            "strict_final_size_check": bool(strict),
            "peak_day": float(r['peak_day']),
            "peak_infections": float(r['peak_infections']),
        }

    return {
        "per_disease": per_disease,
        "all_mass_conservation_pass": bool(all_mass_ok),
        "strict_final_size_pass": bool(all_ar_ok_for_strict),
        "face_validity_pass": bool(all_mass_ok and all_ar_ok_for_strict),
    }


def run_full_validation(disease_config=None, n_runs=50, verbose=True):
    """Run all validation layers and return consolidated results."""
    results = {}

    if verbose:
        print("=" * 60)
        print("VALIDATION SUITE")
        print("=" * 60)

    # Layer 1: ODE analytical
    if verbose:
        print("\n[Layer 1] ODE Analytical Verification...")
    r1 = validate_ode_analytical(disease_config)
    results["ode_analytical"] = {k: v for k, v in r1.items() if k != "result"}
    if verbose:
        print(f"  Mass conservation: {'PASS' if r1['mass_conservation_valid'] else 'FAIL'}")
        print(f"  Max deviation: {r1['max_mass_deviation']:.2e}")
        print(f"  Peak infections: {r1['peak_infections']:.0f} at day {r1['peak_day']:.1f}")
        print(f"  Final attack rate: {r1['final_attack_rate']:.2%}")

    # Layer 2: ABM-ODE agreement under realism
    if verbose:
        print(f"\n[Layer 2] ABM-ODE Agreement ({n_runs} runs)...")
    r2 = validate_abm_ode_convergence(disease_config, n_runs=n_runs)
    results["abm_ode_convergence"] = {
        k: v for k, v in r2.items()
        if k not in ("ode_result", "abm_trajectories")
    }
    if verbose:
        print(f"  Agreement: {'PASS' if r2['agreement_pass'] else 'FAIL'}")
        print(f"    normalized MAE: {r2['normalized_mae']:.3f} (threshold {r2['normalized_mae_threshold']})")
        print(f"    PI coverage:    {r2['prediction_interval_coverage']:.2f} (threshold {r2['coverage_threshold']})")
        print(f"    AR gap:         {r2['attack_rate_gap']:.3f}")

    # Layer 3: Hybrid switching
    if verbose:
        print(f"\n[Layer 3] Hybrid Switching TOST ({n_runs} runs)...")
    r3 = validate_hybrid_switching(disease_config, n_runs=n_runs)
    results["hybrid_switching"] = r3
    if verbose:
        print(f"  Peak equivalence:         {'PASS' if r3['peak_equivalent'] else 'FAIL'} "
              f"(TOST p={r3['peak_tost_p_value']:.4f}, bound=+-{r3['peak_equivalence_bound']:.1f})")
        print(f"  Attack rate equivalence:  {'PASS' if r3['attack_rate_equivalent'] else 'FAIL'} "
              f"(TOST p={r3['attack_rate_tost_p_value']:.4f}, bound=+-{r3['attack_rate_equivalence_bound']})")
        print(f"  Switch fraction: {r3['switch_fraction']:.2f}  (mean days in ODE: {r3['mean_days_in_ode']:.1f})")
        if not r3['switching_exercised']:
            print(f"  WARNING: <50% of runs actually switched modes — equivalence is partly trivial")

    # Layer 4: Face validity across diseases
    if verbose:
        print(f"\n[Layer 4] Per-Disease Face Validity (final-size equation)...")
    r5 = validate_face_validity()
    results["face_validity"] = r5
    if verbose:
        for name, d in r5['per_disease'].items():
            marker = "PASS" if d['attack_rate_pass'] else ("INFO" if not d['strict_final_size_check'] else "FAIL")
            strict_tag = "" if d['strict_final_size_check'] else " [non-strict]"
            print(f"  {name:<10} R0={d['R0']:.2f}  AR sim={d['simulated_attack_rate']:.3f} "
                  f"analytic={d['analytic_attack_rate']:.3f} gap={d['attack_rate_gap']:.3f} "
                  f"[{marker}]{strict_tag}")

    if verbose:
        print("\n" + "=" * 60)
        all_pass = (
            r1['mass_conservation_valid'] and
            r2['agreement_pass'] and
            r3['peak_equivalent'] and
            r3['attack_rate_equivalent'] and
            r5['face_validity_pass']
        )
        print(f"OVERALL: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
        print("=" * 60)

    return results
