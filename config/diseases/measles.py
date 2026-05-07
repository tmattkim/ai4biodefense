"""Measles disease configuration (2025 Texas outbreak).

Sources:
- R0: 12-18 (most contagious known disease), midpoint ~15
- Incubation: 10-14 days, mean ~12 days
- Infectious period: 4-8 days (from 4 days before to 4 days after rash onset), mean ~8
- Serial interval: 11-14 days
- Texas DSHS 2025 outbreak: 762 cases, 99 hospitalizations, 2 deaths
- CDC measles surveillance and modeling reports
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MeaslesConfig:
    name: str = "measles"
    display_name: str = "Measles (2025 Texas)"

    R0: float = 15.0
    incubation_period: float = 12.0
    infectious_period: float = 8.0
    serial_interval: float = 12.0

    initial_infected: int = 5
    initial_exposed: int = 10

    # Measles is transmissible 4 days before rash (presymptomatic)
    presymptomatic_fraction: float = 0.50

    # Moderate overdispersion (Lloyd-Smith et al. 2005)
    dispersion_k: float = 0.5

    @property
    def sigma(self) -> float:
        return 1.0 / self.incubation_period

    @property
    def gamma(self) -> float:
        return 1.0 / self.infectious_period

    @property
    def beta(self) -> float:
        return self.R0 * self.gamma

    # CFR: ~0.1-0.2% in developed countries, higher in young children and adults
    ifr_by_age: tuple = (0.003, 0.001, 0.001, 0.003, 0.006)

    # Hospitalization: ~13% in Texas 2025 (99/762)
    hosp_rate_by_age: tuple = (0.20, 0.10, 0.08, 0.15, 0.25)

    # Measles is almost always symptomatic
    symptomatic_fraction_by_age: tuple = (0.95, 0.95, 0.90, 0.95, 0.95)

    severity_log_mean: float = 0.8
    severity_log_std: float = 0.6

    # Vaccination — critical parameter for measles
    # Texas 2025: ~85% MMR coverage but clustered non-vaccination
    vaccine_coverage: float = 0.85
    vaccine_efficacy_infection: float = 0.97  # 2-dose MMR
    vaccine_efficacy_severe: float = 0.99

    # Contact network scaling — R0=15 requires much higher effective contacts
    contact_scaling_factor: float = 3.0  # Multiplied to community contact weights
