"""Hybrid engine coupling ABM and ODE modes with temporal switching.

Switches between ABM (low prevalence, stochastic effects matter) and
ODE (high prevalence, law of large numbers, computational efficiency).

Thresholds with hysteresis:
- ABM mode when prevalence < 0.1% (captures stochastic extinction, superspreading)
- ODE mode when prevalence > 20% (efficient, LLN makes ODE accurate)
- Wide band prevents rapid oscillation between modes
"""

import numpy as np
from models.seir_ode import solve_seir
from models.abm_engine import ABMEngine
from core.agent import EpiState
from core.contact_network import get_polymod_matrix


class HybridEngine:
    """Orchestrates temporal switching between ABM and compartmental ODE modes.

    The engine starts in ABM mode and switches based on prevalence thresholds.
    State synchronization preserves conservation laws at switching points.
    """

    def __init__(self, agents, networks, disease_config, sim_config, rng,
                 intervention=None):
        """Initialize hybrid engine.

        Args:
            agents: List of Agent objects.
            networks: Dict of contact network layers.
            disease_config: Disease configuration.
            sim_config: SimulationConfig.
            rng: NumPy random generator.
            intervention: Optional Intervention object for ABM self-detection.
        """
        self.disease_config = disease_config
        self.sim_config = sim_config
        self.rng = rng
        self.intervention = intervention
        self.contact_matrix = get_polymod_matrix()
        self.dt = sim_config.get_dt(disease_config.name)

        # Start in ABM mode
        self.mode = "ABM"
        self.abm = ABMEngine(agents, networks, disease_config, self.dt, rng)
        self.n = len(agents)

        # ODE state (populated on switch)
        self.ode_state = None
        self.ode_day_start = None

        # Population counts by age (static)
        self.pop_by_age = np.zeros(5, dtype=float)
        for a in agents:
            self.pop_by_age[a.age_bin] += 1

        # Switching thresholds
        self.abm_to_ode_threshold = sim_config.abm_to_ode_prevalence
        self.ode_to_abm_threshold = sim_config.ode_to_abm_prevalence

        # Cooldown: minimum days between switches to prevent oscillation
        self.last_switch_day = -10.0
        self.switch_cooldown = 5.0

        # History
        self.history = []
        self.mode_log = []

    def step(self, day: float, isolation_decisions: np.ndarray = None,
             intervention_modifier=None, delta: float = 0.0) -> dict:
        """Execute one time step of the hybrid model.

        Args:
            day: Current simulation day.
            isolation_decisions: Boolean array for ABM agent isolation.
            intervention_modifier: Callable(t, beta) -> modified_beta for ODE.
            delta: Detection/isolation rate for ODE Iiso compartment.

        Returns:
            Dict with step results (unified format regardless of mode).
        """
        # Check for mode switching
        self._check_switch(day)
        self.mode_log.append((day, self.mode))

        # Dynamically update delta from intervention if available
        if self.intervention is not None and self.mode == "ODE":
            # Use total infected (I + Iiso) for prevalence, consistent with ABM
            prevalence = (self.ode_state['I'].sum() + self.ode_state['Iiso'].sum()) / self.n if self.ode_state else 0
            delta = self.intervention.get_detection_rate(day, prevalence)

        if self.mode == "ABM":
            result = self._step_abm(day, isolation_decisions)
        else:
            result = self._step_ode(day, intervention_modifier, delta)

        result["mode"] = self.mode
        self.history.append(result)
        return result

    def _step_abm(self, day, isolation_decisions):
        """Run one ABM step."""
        return self.abm.step(day, isolation_decisions, self.intervention)

    def _step_ode(self, day, intervention_modifier, delta):
        """Run one ODE step from current state."""
        # Build prevalence-aware intervention modifier if intervention available
        if self.intervention is not None and self.ode_state is not None:
            # Use total infected (I + Iiso) for prevalence, consistent with ABM
            prevalence = float((self.ode_state['I'].sum() + self.ode_state['Iiso'].sum()) / self.n)
            intervention_ref = self.intervention
            def intervention_modifier(t, beta):
                return intervention_ref.modify_beta(beta, t, prevalence)

        # Build countermeasure effect callable if intervention provides it
        cm_fn = None
        if self.intervention is not None:
            intervention_ref = self.intervention
            def cm_fn(t):
                return intervention_ref.get_countermeasure_effect(int(t))

        # Solve ODE for one timestep
        sol = solve_seir(
            self.disease_config,
            self.pop_by_age,
            initial_conditions=self.ode_state,
            t_span=(day, day + self.dt),
            contact_matrix=self.contact_matrix,
            delta=delta,
            intervention_modifier=intervention_modifier,
            countermeasure_effect_fn=cm_fn,
            dt_eval=self.dt,
        )

        # Update ODE state to end of step
        end_idx = -1  # Last time point
        self.ode_state = {
            'S': sol['S'][end_idx],
            'E': sol['E'][end_idx],
            'I': sol['I'][end_idx],
            'R': sol['R'][end_idx],
            'Iiso': sol['Iiso'][end_idx],
        }

        # Compute state counts for unified output
        S_total = float(sol['S_total'][end_idx])
        E_total = float(sol['E_total'][end_idx])
        I_total = float(sol['I_total'][end_idx])
        R_total = float(sol['R_total'][end_idx])
        Iiso_total = float(sol['Iiso_total'][end_idx])

        # Report I + Iiso as total infected for consistency with ABM mode,
        # where state_counts[2] counts ALL EpiState.INFECTED (including isolated)
        total_infected = I_total + Iiso_total
        state_counts = np.array([S_total, E_total, total_infected, R_total])
        prevalence = total_infected / self.n

        return {
            "day": day,
            "new_infections": 0,   # Not tracked in ODE mode
            "new_infectious": 0,
            "new_recoveries": 0,
            "new_deaths": 0,
            "new_hospitalizations": 0,
            "state_counts": state_counts,
            "state_counts_by_age": np.vstack([
                sol['S'][end_idx], sol['E'][end_idx],
                sol['I'][end_idx], sol['R'][end_idx],
            ]),
            "prevalence": prevalence,
            "isolation_rate": Iiso_total / max(I_total + Iiso_total, 1),
            "transmission_events": [],
        }

    def _check_switch(self, day):
        """Check if mode should switch based on prevalence thresholds."""
        if day - self.last_switch_day < self.switch_cooldown:
            return

        if self.mode == "ABM":
            prevalence = np.sum(
                self.abm.states == EpiState.INFECTED
            ) / self.n
            if prevalence > self.abm_to_ode_threshold:
                self._switch_abm_to_ode(day)
        else:
            # Use I + Iiso for total infected (I is non-isolated only in ODE)
            I_total = self.ode_state['I'].sum() + self.ode_state['Iiso'].sum()
            prevalence = I_total / self.n
            if prevalence < self.ode_to_abm_threshold:
                self._switch_ode_to_abm(day)

    def _switch_abm_to_ode(self, day):
        """Aggregate ABM agent states into ODE compartments.

        Vaccination correction: vaccinated susceptibles have reduced
        susceptibility in ABM (leaky vaccine). For ODE equivalence,
        move the effectively-immune fraction to R.
        """
        counts = self.abm.get_state_counts_by_age()
        S = counts[0].astype(float)
        R = counts[3].astype(float)

        # Correct for vaccination: move effectively-immune susceptibles to R
        vaccine_eff = getattr(self.disease_config, 'vaccine_efficacy_infection', 0.0)
        if vaccine_eff > 0:
            for ab in range(5):
                ab_mask = (self.abm.age_bins == ab)
                s_mask = ab_mask & (self.abm.states == EpiState.SUSCEPTIBLE)
                n_vacc_s = int(np.sum(s_mask & self.abm.is_vaccinated))
                # Effectively immune = vaccinated * efficacy
                immune_transfer = n_vacc_s * vaccine_eff
                S[ab] -= immune_transfer
                R[ab] += immune_transfer

        Iiso = counts[4].astype(float)
        self.ode_state = {
            'S': S,
            'E': counts[1].astype(float),
            'I': counts[2].astype(float) - Iiso,  # Non-isolated infected only
            'R': R,
            'Iiso': Iiso,
        }
        self.ode_day_start = day
        self.mode = "ODE"
        self.last_switch_day = day

    def _switch_ode_to_abm(self, day):
        """Distribute ODE state back to ABM agents."""
        state_array = np.array([
            self.ode_state['S'],
            self.ode_state['E'],
            self.ode_state['I'],
            self.ode_state['R'],
            self.ode_state['Iiso'],
        ])
        self.abm.sync_from_ode(state_array, self.rng)
        self.mode = "ABM"
        self.last_switch_day = day

    def get_trajectory(self):
        """Return the full simulation trajectory as arrays.

        Returns:
            Dict with keys: days, S, E, I, R, prevalence, mode
        """
        if not self.history:
            return None

        days = np.array([h['day'] for h in self.history])
        state_counts = np.array([h['state_counts'] for h in self.history])
        prevalence = np.array([h['prevalence'] for h in self.history])
        modes = [h['mode'] for h in self.history]

        return {
            'days': days,
            'S': state_counts[:, 0],
            'E': state_counts[:, 1],
            'I': state_counts[:, 2],
            'R': state_counts[:, 3],
            'prevalence': prevalence,
            'mode': modes,
        }

    def run(self, time_horizon: int = None, isolation_engine=None,
            intervention_modifier=None, delta: float = 0.0,
            progress_callback=None) -> dict:
        """Run the complete simulation.

        Args:
            time_horizon: Number of days to simulate.
            isolation_engine: Optional async callable(abm, day) -> isolation_decisions.
            intervention_modifier: Optional callable(t, beta) -> beta for ODE mode.
            delta: Detection rate for ODE Iiso compartment.
            progress_callback: Optional callable(day, result) for progress reporting.

        Returns:
            Full trajectory dict.
        """
        if time_horizon is None:
            time_horizon = self.sim_config.get_time_horizon(self.disease_config.name)

        days = np.arange(0, time_horizon, self.dt)

        for day in days:
            # Get isolation decisions if in ABM mode and engine provided
            isolation_decisions = None
            if self.mode == "ABM" and isolation_engine is not None:
                isolation_decisions = isolation_engine(self.abm, day)

            result = self.step(
                day, isolation_decisions, intervention_modifier, delta
            )

            if progress_callback:
                progress_callback(day, result)

            # Early termination if epidemic is over
            if (result['state_counts'][1] + result['state_counts'][2]) < 1:
                break

        return self.get_trajectory()
