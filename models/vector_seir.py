"""Coupled human-vector SEIR model for dengue (Ross-Macdonald extension).

Human compartments: S_h, E_h, I_h, R_h
Mosquito compartments: S_m, E_m, I_m (no recovery — mosquitoes die infected)

Human dynamics:
    dS_h/dt = -a * b * I_m/N_h * S_h
    dE_h/dt = a * b * I_m/N_h * S_h - sigma_h * E_h
    dI_h/dt = sigma_h * E_h - gamma_h * I_h
    dR_h/dt = gamma_h * I_h

Mosquito dynamics:
    dS_m/dt = mu_m * N_m - a * c * I_h/N_h * S_m - mu_m * S_m
    dE_m/dt = a * c * I_h/N_h * S_m - (sigma_m + mu_m) * E_m
    dI_m/dt = sigma_m * E_m - mu_m * I_m

where:
    a = biting rate
    b = P(human infected | bite by infected mosquito)
    c = P(mosquito infected | bite on viremic human)
    sigma_h = 1/incubation_human
    sigma_m = 1/extrinsic_incubation
    gamma_h = 1/infectious_period
    mu_m = mosquito mortality rate
    N_m = total mosquito population
"""

import numpy as np
from scipy.integrate import solve_ivp


def vector_seir_derivatives(t, y, params):
    """Compute derivatives for coupled human-vector SEIR system.

    y = [S_h, E_h, I_h, R_h, S_m, E_m, I_m]
    """
    S_h, E_h, I_h, R_h, S_m, E_m, I_m = y

    N_h = S_h + E_h + I_h + R_h
    N_m = S_m + E_m + I_m

    a = params["biting_rate"]
    b = params["mosquito_to_human_prob"]
    c = params["human_to_mosquito_prob"]
    sigma_h = params["sigma_h"]
    sigma_m = params["sigma_m"]
    gamma_h = params["gamma_h"]
    mu_m = params["mu_m"]

    # Human dynamics
    force_human = a * b * I_m / max(N_h, 1)
    dS_h = -force_human * S_h
    dE_h = force_human * S_h - sigma_h * E_h
    dI_h = sigma_h * E_h - gamma_h * I_h
    dR_h = gamma_h * I_h

    # Mosquito dynamics
    force_mosquito = a * c * I_h / max(N_h, 1)
    dS_m = mu_m * N_m - force_mosquito * S_m - mu_m * S_m
    dE_m = force_mosquito * S_m - (sigma_m + mu_m) * E_m
    dI_m = sigma_m * E_m - mu_m * I_m

    return [dS_h, dE_h, dI_h, dR_h, dS_m, dE_m, dI_m]


def solve_vector_seir(
    dengue_config,
    human_population: int = 10000,
    t_span: tuple = None,
    dt_eval: float = 0.25,
):
    """Solve the coupled human-vector SEIR system.

    Args:
        dengue_config: DengueConfig object.
        human_population: Total human population.
        t_span: (t_start, t_end).
        dt_eval: Output time step.

    Returns:
        Dict with time series for all compartments.
    """
    N_h = float(human_population)
    N_m = N_h * dengue_config.mosquito_per_human_ratio

    if t_span is None:
        t_span = (0, 365)

    # Initial conditions
    I_h0 = float(dengue_config.initial_infected)
    E_h0 = float(dengue_config.initial_exposed)
    S_h0 = N_h - I_h0 - E_h0
    R_h0 = 0.0

    # Start with all mosquitoes susceptible, some exposed
    I_m0 = 5.0  # A few infected mosquitoes to seed
    E_m0 = 10.0
    S_m0 = N_m - I_m0 - E_m0

    y0 = [S_h0, E_h0, I_h0, R_h0, S_m0, E_m0, I_m0]

    params = {
        "biting_rate": dengue_config.mosquito_biting_rate,
        "mosquito_to_human_prob": dengue_config.mosquito_to_human_probability,
        "human_to_mosquito_prob": dengue_config.mosquito_infection_probability,
        "sigma_h": dengue_config.sigma,
        "sigma_m": 1.0 / dengue_config.mosquito_extrinsic_incubation,
        "gamma_h": dengue_config.gamma,
        "mu_m": dengue_config.mosquito_mortality_rate,
    }

    t_eval = np.arange(t_span[0], t_span[1] + dt_eval, dt_eval)

    sol = solve_ivp(
        fun=lambda t, y: vector_seir_derivatives(t, y, params),
        t_span=t_span,
        y0=y0,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol.success:
        raise RuntimeError(f"Vector SEIR solver failed: {sol.message}")

    return {
        "t": sol.t,
        "S_h": sol.y[0],
        "E_h": sol.y[1],
        "I_h": sol.y[2],
        "R_h": sol.y[3],
        "S_m": sol.y[4],
        "E_m": sol.y[5],
        "I_m": sol.y[6],
        # Unified interface
        "S_total": sol.y[0],
        "E_total": sol.y[1],
        "I_total": sol.y[2],
        "R_total": sol.y[3],
    }
