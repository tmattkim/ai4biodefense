"""Age-stratified SEIR ODE solver with optional isolated compartment.

Implements the compartmental component of the hybrid model:
    dS_i/dt = -S_i * sum_j(beta * C_ij * I_j / N_j)
    dE_i/dt = S_i * sum_j(beta * C_ij * I_j / N_j) - sigma * E_i
    dI_i/dt = sigma * E_i - gamma * I_i - delta * I_i
    dR_i/dt = gamma * I_i + gamma * Iiso_i
    dIiso_i/dt = delta * I_i - gamma * Iiso_i

where delta is the detection/isolation rate (0 in baseline, increased by system AI).
Isolated individuals transmit at reduced rate (epsilon * beta).
"""

import numpy as np
from scipy.integrate import solve_ivp
from core.contact_network import get_polymod_matrix


def seir_derivatives(t, y, params):
    """Compute derivatives for age-stratified SEIR+Iiso system.

    Args:
        t: Current time (unused for autonomous system, but needed by solver).
        y: State vector, flat array of length 5 * n_age_bins:
           [S0..S4, E0..E4, I0..I4, R0..R4, Iiso0..Iiso4]
        params: Dict with keys:
           beta, sigma, gamma, delta (detection rate),
           epsilon (isolation leakage, default 0.1),
           contact_matrix (5x5), N_by_age (array of 5),
           intervention_modifier (optional callable(t, params) -> modified beta)

    Returns:
        Flat derivative array of same shape as y.
    """
    n = params["n_age_bins"]
    S = y[0*n:1*n]
    E = y[1*n:2*n]
    I = y[2*n:3*n]
    R = y[3*n:4*n]
    Iiso = y[4*n:5*n]

    beta = params["beta"]
    sigma = params["sigma"]
    gamma = params["gamma"]
    delta = params.get("delta", 0.0)
    epsilon = params.get("epsilon", 0.1)  # Isolation reduces transmission to 10%
    C = params["contact_matrix"]
    N_by_age = params["N_by_age"]

    # Seasonal forcing (e.g., influenza)
    seasonal_amp = params.get("seasonal_amplitude", 0.0)
    if seasonal_amp > 0:
        peak_day = params.get("seasonal_peak_day", 0)
        beta *= 1.0 + seasonal_amp * np.cos(2.0 * np.pi * (t - peak_day) / 365.0)

    # Apply time-varying intervention modifier if present
    modifier = params.get("intervention_modifier")
    if modifier is not None:
        beta = modifier(t, beta)

    # Apply countermeasure effect (AI-accelerated biomanufacturing)
    # Increases recovery rate as treatments become available
    cm_fn = params.get("countermeasure_effect_fn")
    if cm_fn is not None:
        cm_effect = cm_fn(t)
        gamma = gamma * (1.0 + cm_effect)

    # Force of infection per age group: lambda_i = beta * sum_j(C_ij * I_j / N_j)
    # Include leaked transmission from isolated compartment
    effective_I = I + epsilon * Iiso
    infection_pressure = effective_I / np.maximum(N_by_age, 1.0)
    force_of_infection = beta * C @ infection_pressure

    dS = -S * force_of_infection
    dE = S * force_of_infection - sigma * E
    dI = sigma * E - gamma * I - delta * I
    dR = gamma * I + gamma * Iiso
    dIiso = delta * I - gamma * Iiso

    return np.concatenate([dS, dE, dI, dR, dIiso])


def solve_seir(
    disease_config,
    population_counts_by_age,
    initial_conditions=None,
    t_span=None,
    contact_matrix=None,
    delta=0.0,
    epsilon=0.1,
    intervention_modifier=None,
    countermeasure_effect_fn=None,
    dt_eval=1.0,
):
    """Solve the age-stratified SEIR+Iiso system.

    Args:
        disease_config: Disease configuration object with beta, sigma, gamma.
        population_counts_by_age: Array of population in each age bin.
        initial_conditions: Dict with 'S', 'E', 'I', 'R', 'Iiso' arrays (per age).
                           If None, uses disease_config defaults.
        t_span: (t_start, t_end) tuple. If None, uses (0, time_horizon).
        contact_matrix: 5x5 contact matrix. If None, uses POLYMOD.
        delta: Detection/isolation rate (0 for baseline).
        epsilon: Transmission reduction for isolated (0.1 = 90% reduction).
        intervention_modifier: Optional callable(t, beta) -> modified_beta.
        dt_eval: Time step for output evaluation points.

    Returns:
        Dict with keys:
            t: time array
            S, E, I, R, Iiso: arrays of shape (n_times, n_age_bins)
            S_total, E_total, I_total, R_total, Iiso_total: summed across ages
    """
    n_age = len(population_counts_by_age)
    N_by_age = np.array(population_counts_by_age, dtype=float)
    N_total = N_by_age.sum()

    if contact_matrix is None:
        contact_matrix = get_polymod_matrix()

    if t_span is None:
        from config.base import SimulationConfig
        horizon = SimulationConfig().get_time_horizon(disease_config.name)
        t_span = (0, horizon)

    # Initial conditions
    if initial_conditions is None:
        S0 = N_by_age.copy()
        E0 = np.zeros(n_age)
        I0 = np.zeros(n_age)
        R0 = np.zeros(n_age)
        Iiso0 = np.zeros(n_age)

        # Distribute initial cases proportionally across age groups
        n_init_I = getattr(disease_config, 'initial_infected', 10)
        n_init_E = getattr(disease_config, 'initial_exposed', 0)
        fracs = N_by_age / N_total
        I0 = np.round(fracs * n_init_I).astype(float)
        E0 = np.round(fracs * n_init_E).astype(float)
        S0 -= (I0 + E0)
    else:
        S0 = np.array(initial_conditions['S'], dtype=float)
        E0 = np.array(initial_conditions['E'], dtype=float)
        I0 = np.array(initial_conditions['I'], dtype=float)
        R0 = np.array(initial_conditions['R'], dtype=float)
        Iiso0 = np.array(initial_conditions.get('Iiso', np.zeros(n_age)), dtype=float)

    y0 = np.concatenate([S0, E0, I0, R0, Iiso0])

    params = {
        "n_age_bins": n_age,
        "beta": disease_config.beta,
        "sigma": disease_config.sigma,
        "gamma": disease_config.gamma,
        "delta": delta,
        "epsilon": epsilon,
        "contact_matrix": contact_matrix,
        "N_by_age": N_by_age,
        "intervention_modifier": intervention_modifier,
        "countermeasure_effect_fn": countermeasure_effect_fn,
        "seasonal_amplitude": getattr(disease_config, 'seasonal_amplitude', 0.0),
        "seasonal_peak_day": getattr(disease_config, 'seasonal_peak_day', 0),
    }

    t_eval = np.arange(t_span[0], t_span[1] + dt_eval, dt_eval)

    sol = solve_ivp(
        fun=lambda t, y: seir_derivatives(t, y, params),
        t_span=t_span,
        y0=y0,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
        max_step=0.5,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    result = {
        "t": sol.t,
        "S": sol.y[0*n_age:1*n_age].T,
        "E": sol.y[1*n_age:2*n_age].T,
        "I": sol.y[2*n_age:3*n_age].T,
        "R": sol.y[3*n_age:4*n_age].T,
        "Iiso": sol.y[4*n_age:5*n_age].T,
    }

    # Aggregate across age groups
    result["S_total"] = result["S"].sum(axis=1)
    result["E_total"] = result["E"].sum(axis=1)
    result["I_total"] = result["I"].sum(axis=1)
    result["R_total"] = result["R"].sum(axis=1)
    result["Iiso_total"] = result["Iiso"].sum(axis=1)

    return result


def verify_mass_conservation(result, N_total, tol=1e-4):
    """Verify S + E + I + R + Iiso = N at all time steps.

    Returns:
        (is_valid, max_deviation)
    """
    total = (
        result["S_total"] + result["E_total"] + result["I_total"]
        + result["R_total"] + result["Iiso_total"]
    )
    max_dev = np.max(np.abs(total - N_total))
    return max_dev < tol, max_dev


def compute_R_effective(result, disease_config, population_counts_by_age, contact_matrix=None):
    """Compute time-varying effective reproduction number.

    R_eff(t) = R0 * S_total(t) / N_total
    (simplified; age-structured version would use next-generation matrix)
    """
    if contact_matrix is None:
        contact_matrix = get_polymod_matrix()
    N_total = np.sum(population_counts_by_age)
    R_eff = disease_config.R0 * result["S_total"] / N_total
    return R_eff
