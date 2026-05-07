"""COVID-19 (Delta variant) disease configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class COVID19Config:
    """Epidemiological parameters for COVID-19 Delta variant.

    Sources:
    - R0: Estimated 3.5-6.0 for Delta; we use 4.0 as midpoint
    - Incubation: Lauer et al. 2020, mean 5.2 days
    - Infectious period: He et al. 2020, ~10 days
    - Serial interval: ~5.2 days (He et al. 2020)
    - IFR/hospitalization by age: CDC COVID-19 data
    """

    name: str = "covid19"
    display_name: str = "COVID-19 (Delta)"

    # Core SEIR parameters
    R0: float = 4.0
    incubation_period: float = 5.2     # days (1/sigma)
    infectious_period: float = 10.0    # days (1/gamma)
    serial_interval: float = 5.2      # days

    # Initial conditions (per 10,000 population)
    initial_infected: int = 10
    initial_exposed: int = 30

    # Presymptomatic transmission fraction
    presymptomatic_fraction: float = 0.35

    # Overdispersion: k=0.1 means ~10% of cases cause ~80% of infections
    # Lancet Infect Dis 2021; lower k = more superspreading
    dispersion_k: float = 0.1

    @property
    def sigma(self) -> float:
        """Rate of progression from Exposed to Infected (1/incubation)."""
        return 1.0 / self.incubation_period

    @property
    def gamma(self) -> float:
        """Recovery rate (1/infectious period)."""
        return 1.0 / self.infectious_period

    @property
    def beta(self) -> float:
        """Transmission rate derived from R0 = beta / gamma."""
        return self.R0 * self.gamma

    # Age-stratified infection fatality rate (per age bin: 0-4, 5-17, 18-49, 50-64, 65+)
    ifr_by_age: tuple = (0.00003, 0.0001, 0.002, 0.013, 0.09)

    # Hospitalization rate by age bin
    hosp_rate_by_age: tuple = (0.001, 0.003, 0.04, 0.10, 0.27)

    # Probability of being symptomatic by age bin
    symptomatic_fraction_by_age: tuple = (0.50, 0.55, 0.65, 0.75, 0.85)

    # Symptom severity distribution parameters (lognormal: mean, std of log)
    severity_log_mean: float = 0.5
    severity_log_std: float = 0.8

    # Vaccination (Delta era ~50% US coverage, variable efficacy)
    vaccine_coverage: float = 0.0  # Set to 0 for unvaccinated scenario
    vaccine_efficacy_infection: float = 0.60
    vaccine_efficacy_severe: float = 0.90
