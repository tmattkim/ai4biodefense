"""Counterfactual analysis with shared epidemic infrastructure.

All 4 intervention scenarios share:
- Same population (demographics, archetypes, households)
- Same contact network
- Same base random seed for infection draws

This ensures differences in outcomes are attributable to interventions,
not stochastic variation between independent simulations.
"""

import hashlib
import numpy as np
from config.base import SimulationConfig
from core.population import create_population, assign_ai_access
from core.contact_network import build_contact_network
from models.hybrid_engine import HybridEngine
from analysis.metrics import compute_metrics, compute_comparative_metrics


class CounterfactualAnalyzer:
    """Run counterfactual intervention comparisons with shared infrastructure."""

    def __init__(self, disease_config, sim_config=None):
        self.disease_config = disease_config
        self.sim_config = sim_config or SimulationConfig()

    def run_counterfactual_set(
        self, replication: int, scenarios: dict, verbose: bool = False,
    ) -> dict:
        """Run all scenarios with shared population and network.

        Args:
            replication: Replication index (used to derive seed).
            scenarios: Dict mapping scenario_name -> Scenario object.
            verbose: Print progress.

        Returns:
            Dict mapping scenario_name -> {trajectory, metrics, engine_telemetry}
        """
        base_seed = self.sim_config.seed + replication

        # Generate shared population
        rng_pop = np.random.default_rng(base_seed)
        agents = create_population(self.sim_config, self.disease_config, rng_pop)

        # Generate shared contact network
        rng_net = np.random.default_rng(base_seed + 100000)
        networks = build_contact_network(agents, rng_net)

        results = {}

        for scenario_name, scenario in scenarios.items():
            intervention = scenario.intervention
            if verbose:
                interv_name = intervention.get_name()
                print(f"  Running {scenario_name} [{interv_name}]...")

            # Fresh RNG for this scenario (but same population/network)
            # Use deterministic hash (Python's hash() varies across sessions)
            name_hash = int(hashlib.sha256(scenario_name.encode()).hexdigest()[:8], 16)
            rng_sim = np.random.default_rng(base_seed + name_hash % 100000)

            # Reset intervention state
            if hasattr(intervention, 'reset'):
                intervention.reset()

            # Assign AI access based on intervention
            ai_rate = intervention.get_ai_tool_access_rate()
            rng_ai = np.random.default_rng(base_seed + 200000)
            # Re-create agents for fresh state (but same demographics)
            agents_copy = create_population(self.sim_config, self.disease_config, np.random.default_rng(base_seed))
            if ai_rate > 0:
                assign_ai_access(agents_copy, ai_rate, rng_ai)

            # Run simulation
            engine = HybridEngine(
                agents_copy, networks, self.disease_config,
                self.sim_config, rng_sim,
                intervention=intervention,
            )

            # Intervention modifier for ODE mode
            def make_intervention_modifier(interv):
                def modifier(t, beta):
                    prevalence = 0.05  # Approximate; exact tracking not available in ODE
                    return interv.modify_beta(beta, t, prevalence)
                return modifier

            delta = intervention.get_detection_rate(0, 0.01)

            trajectory = engine.run(
                intervention_modifier=make_intervention_modifier(intervention),
                delta=delta,
            )

            # Compute metrics
            if trajectory is not None:
                metrics = compute_metrics(
                    trajectory, self.disease_config,
                    self.sim_config.population_size,
                )
            else:
                metrics = {}

            # Engine telemetry: mode-log share for hybrid engine. Tells us how
            # much of this run was actually spent in ABM vs ODE — direct evidence
            # that the agents are doing work.
            mode_log = list(getattr(engine, 'mode_log', []))
            n_total = max(len(mode_log), 1)
            n_abm = sum(1 for entry in mode_log if entry[1] == "ABM")
            n_ode = sum(1 for entry in mode_log if entry[1] == "ODE")
            switches = sum(
                1 for i in range(1, len(mode_log))
                if mode_log[i][1] != mode_log[i - 1][1]
            )
            engine_telemetry = {
                "abm_fraction": n_abm / n_total,
                "ode_fraction": n_ode / n_total,
                "n_switches": int(switches),
                "ever_switched_to_ode": bool(n_ode > 0),
                "n_steps_logged": int(len(mode_log)),
            }

            results[scenario_name] = {
                "trajectory": trajectory,
                "metrics": metrics,
                "engine_telemetry": engine_telemetry,
            }

        # Compute comparative metrics
        if "baseline" in results and results["baseline"]["metrics"]:
            baseline_metrics = results["baseline"]["metrics"]
            for scenario_name in results:
                if scenario_name == "baseline":
                    continue
                if results[scenario_name]["metrics"]:
                    results[scenario_name]["comparative"] = compute_comparative_metrics(
                        baseline_metrics, results[scenario_name]["metrics"]
                    )

        return results
