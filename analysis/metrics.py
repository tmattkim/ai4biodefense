"""Outcome metric calculations for simulation results."""

import numpy as np


def compute_metrics(trajectory: dict, disease_config, population_size: int) -> dict:
    """Compute all outcome metrics from a single simulation run.

    Args:
        trajectory: Dict from HybridEngine.get_trajectory() with keys:
                    days, S, E, I, R, prevalence
        disease_config: Disease configuration for severity/IFR parameters.
        population_size: Total population N.

    Returns:
        Dict of outcome metrics.
    """
    days = trajectory['days']
    S = trajectory['S']
    E = trajectory['E']
    I = trajectory['I']
    R = trajectory['R']

    # Cumulative attack rate
    final_R = R[-1] if len(R) > 0 else 0
    cumulative_attack_rate = final_R / population_size

    # Peak incidence
    peak_idx = np.argmax(I)
    peak_incidence = float(I[peak_idx])
    peak_timing = float(days[peak_idx])

    # Epidemic duration (days from first case to <1 active case)
    active = E + I
    above_threshold = np.where(active >= 1.0)[0]
    if len(above_threshold) > 0:
        epidemic_duration = float(days[above_threshold[-1]] - days[above_threshold[0]])
    else:
        epidemic_duration = 0.0

    # Healthcare burden estimate
    # Hospital-days = sum over time of hospitalized agents
    # Approximate: peak_incidence * mean_hosp_rate * mean_hospital_stay
    mean_hosp_rate = np.mean(disease_config.hosp_rate_by_age)
    hospital_days = float(np.sum(I * mean_hosp_rate * 1.0))  # Rough estimate

    # R_effective trajectory
    R_eff = disease_config.R0 * S / population_size

    return {
        "cumulative_attack_rate": float(cumulative_attack_rate),
        "peak_incidence": float(peak_incidence),
        "peak_timing": float(peak_timing),
        "epidemic_duration": float(epidemic_duration),
        "hospital_days": float(hospital_days),
        "total_infections": float(final_R),
        "R_eff_trajectory": R_eff.tolist() if hasattr(R_eff, 'tolist') else list(R_eff),
        "R_eff_final": float(R_eff[-1]) if len(R_eff) > 0 else 0.0,
    }


def compute_comparative_metrics(baseline: dict, intervention: dict) -> dict:
    """Compute relative effectiveness metrics.

    Args:
        baseline: Metrics dict from baseline scenario.
        intervention: Metrics dict from intervention scenario.

    Returns:
        Dict of comparative metrics.
    """
    def pct_change(base, interv):
        if base == 0:
            return 0.0
        return 100.0 * (base - interv) / base

    return {
        "peak_reduction_pct": pct_change(
            baseline["peak_incidence"], intervention["peak_incidence"]
        ),
        "attack_rate_reduction_pct": pct_change(
            baseline["cumulative_attack_rate"],
            intervention["cumulative_attack_rate"],
        ),
        "peak_delay_days": (
            intervention["peak_timing"] - baseline["peak_timing"]
        ),
        "hospital_burden_reduction_pct": pct_change(
            baseline["hospital_days"], intervention["hospital_days"]
        ),
        "duration_change_days": (
            intervention["epidemic_duration"] - baseline["epidemic_duration"]
        ),
    }
