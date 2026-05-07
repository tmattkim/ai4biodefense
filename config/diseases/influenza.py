"""Influenza disease configuration (seasonal H3N2).

Sources:
- R0: Yang et al. 2015, seasonal flu 1.3-2.1, midpoint ~1.7
- Incubation: 1-4 days, mean ~2 days
- Infectious period: 3-7 days, mean ~5 days
- Serial interval: 2-4 days, mean ~3 days
- CDC FluView and FluSurv-NET for age-stratified data
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class InfluenzaConfig:
    name: str = "influenza"
    display_name: str = "Influenza (H3N2)"

    R0: float = 1.7
    incubation_period: float = 2.0
    infectious_period: float = 5.0
    serial_interval: float = 3.0

    initial_infected: int = 10
    initial_exposed: int = 20

    presymptomatic_fraction: float = 0.30

    @property
    def sigma(self) -> float:
        return 1.0 / self.incubation_period

    @property
    def gamma(self) -> float:
        return 1.0 / self.infectious_period

    @property
    def beta(self) -> float:
        return self.R0 * self.gamma

    # Age-stratified IFR (per age bin: 0-4, 5-17, 18-49, 50-64, 65+)
    ifr_by_age: tuple = (0.0001, 0.00005, 0.0002, 0.001, 0.01)

    # Hospitalization rates
    hosp_rate_by_age: tuple = (0.01, 0.005, 0.008, 0.02, 0.05)

    # Symptomatic fraction
    symptomatic_fraction_by_age: tuple = (0.60, 0.55, 0.50, 0.60, 0.70)

    severity_log_mean: float = 0.3
    severity_log_std: float = 0.7

    # Vaccination (40-50% coverage with ~40% VE for H3N2)
    vaccine_coverage: float = 0.45
    vaccine_efficacy_infection: float = 0.40
    vaccine_efficacy_severe: float = 0.60

    # Seasonal forcing amplitude (beta multiplier oscillation)
    seasonal_amplitude: float = 0.30
    seasonal_peak_day: int = 0  # Day 0 = peak flu season (January)
