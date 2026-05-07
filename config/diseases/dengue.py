"""Dengue disease configuration.

Sources:
- R0: 1.5-6.0 (varies by serotype/setting), midpoint ~3.0
- Human incubation: 4-10 days, mean ~5.9 days
- Viremic (infectious) period: 4-7 days, mean ~5 days
- Serial interval: ~14-20 days (includes mosquito extrinsic incubation)
- Extrinsic incubation in mosquito: 8-12 days, mean ~10 days
- OpenDengue dataset for calibration
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DengueConfig:
    name: str = "dengue"
    display_name: str = "Dengue"

    R0: float = 3.0
    incubation_period: float = 5.9    # Human intrinsic incubation
    infectious_period: float = 5.0    # Viremic period
    serial_interval: float = 17.0     # Includes mosquito extrinsic incubation

    initial_infected: int = 10
    initial_exposed: int = 15

    presymptomatic_fraction: float = 0.10

    @property
    def sigma(self) -> float:
        return 1.0 / self.incubation_period

    @property
    def gamma(self) -> float:
        return 1.0 / self.infectious_period

    @property
    def beta(self) -> float:
        return self.R0 * self.gamma

    # CFR is low (~1-2.5% for severe dengue, <0.1% overall)
    ifr_by_age: tuple = (0.001, 0.001, 0.0005, 0.001, 0.002)

    hosp_rate_by_age: tuple = (0.05, 0.03, 0.02, 0.03, 0.05)

    # ~75% symptomatic (dengue fever), 25% asymptomatic
    symptomatic_fraction_by_age: tuple = (0.75, 0.70, 0.65, 0.70, 0.80)

    severity_log_mean: float = 0.5
    severity_log_std: float = 0.9

    vaccine_coverage: float = 0.0
    vaccine_efficacy_infection: float = 0.0
    vaccine_efficacy_severe: float = 0.0

    # Vector-specific parameters (for Ross-Macdonald extension)
    mosquito_extrinsic_incubation: float = 10.0  # days
    mosquito_biting_rate: float = 0.5             # bites per mosquito per day
    mosquito_infection_probability: float = 0.5    # P(mosquito infected per bite on viremic human)
    mosquito_to_human_probability: float = 0.5     # P(human infected per bite from infected mosquito)
    mosquito_mortality_rate: float = 0.1           # Per day (mean lifespan ~10 days)
    mosquito_per_human_ratio: float = 2.0          # Mosquitoes per human in endemic area
