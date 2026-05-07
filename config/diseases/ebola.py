"""Ebola virus disease configuration (2014-2016 West Africa outbreak).

Sources:
- R0: WHO Ebola Response Team 2014, 1.5-2.5 (West Africa), midpoint ~1.8
- Incubation: 2-21 days, mean ~9.4 days (WHO)
- Infectious period: 4-10 days, mean ~7 days
- Serial interval: ~15 days
- CFR: 50-70% (West Africa 2014, varied by country)
- cmrivers/ebola GitHub dataset
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EbolaConfig:
    name: str = "ebola"
    display_name: str = "Ebola (2014 West Africa)"

    R0: float = 1.8
    incubation_period: float = 9.4
    infectious_period: float = 7.0
    serial_interval: float = 15.0

    initial_infected: int = 5
    initial_exposed: int = 10

    presymptomatic_fraction: float = 0.0  # Ebola not transmissible before symptoms

    # Overdispersion: similar to SARS (Lloyd-Smith et al. 2005)
    dispersion_k: float = 0.18

    @property
    def sigma(self) -> float:
        return 1.0 / self.incubation_period

    @property
    def gamma(self) -> float:
        return 1.0 / self.infectious_period

    @property
    def beta(self) -> float:
        return self.R0 * self.gamma

    # Very high CFR, less age-dependent than respiratory diseases
    ifr_by_age: tuple = (0.55, 0.50, 0.55, 0.60, 0.70)

    # Hospitalization (most cases require hospitalization/isolation)
    hosp_rate_by_age: tuple = (0.80, 0.75, 0.70, 0.75, 0.80)

    # All Ebola cases are symptomatic once infectious
    symptomatic_fraction_by_age: tuple = (1.0, 1.0, 1.0, 1.0, 1.0)

    severity_log_mean: float = 1.5  # Most cases are severe
    severity_log_std: float = 0.5

    vaccine_coverage: float = 0.0
    vaccine_efficacy_infection: float = 0.0
    vaccine_efficacy_severe: float = 0.0

    # Ebola-specific: contact patterns shift toward household/healthcare
    # Higher household transmission, burial transmission
    household_transmission_multiplier: float = 2.0
    community_transmission_multiplier: float = 0.3
    burial_transmission_probability: float = 0.20
