"""Global simulation configuration."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SimulationConfig:
    """Central configuration for all simulation runs."""

    population_size: int = 10_000
    num_replications: int = 100
    seed: int = 42

    # Age bins matching POLYMOD (indices 0-4)
    age_bins: Tuple[Tuple[int, int], ...] = (
        (0, 4),
        (5, 17),
        (18, 49),
        (50, 64),
        (65, 100),
    )

    # US Census ACS age distribution (approximate proportions per bin)
    # 0-4: 6%, 5-17: 16%, 18-49: 39%, 50-64: 19%, 65+: 17% (sums to 0.97; remainder to largest bin)
    age_distribution: Tuple[float, ...] = (0.06, 0.16, 0.39, 0.19, 0.17)

    # Household parameters
    mean_household_size: float = 2.5

    # Hybrid model switching thresholds (prevalence = I/N)
    abm_to_ode_prevalence: float = 0.20
    ode_to_abm_prevalence: float = 0.001

    # LLM query parameters
    llm_sample_rate: float = 0.05       # Fraction of agents queried per query day
    llm_query_interval: int = 3         # Days between LLM queries
    llm_max_concurrent: int = 10        # Max simultaneous API calls

    # Archetype distribution (8 types, must sum to 1.0)
    archetype_distribution: Tuple[float, ...] = (
        0.12,  # Young Invincible
        0.18,  # Working Parent
        0.20,  # Essential Worker
        0.12,  # Elderly Cautious
        0.08,  # Healthcare-Informed
        0.12,  # Distrust/Skeptic
        0.08,  # Immunocompromised/Chronic
        0.10,  # Community-Oriented
    )

    def get_dt(self, disease_name: str) -> float:
        """Variable timestep based on disease serial interval."""
        if disease_name.lower() in ("dengue",):
            return 0.25
        return 1.0

    def get_time_horizon(self, disease_name: str) -> int:
        """Per-disease simulation time horizon in days."""
        horizons = {
            "covid19": 365,
            "influenza": 365,
            "ebola": 365,
            "dengue": 365,
            "measles": 365,
        }
        return horizons.get(disease_name.lower(), 365)
