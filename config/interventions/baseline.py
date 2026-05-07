"""Baseline intervention scenario: standard public health response without AI.

Represents conventional public health infrastructure responding to an outbreak.
After surveillance delay (7 days for manual reporting), a modest public health
response activates with slightly enhanced detection and contact tracing.
"""

from interventions.base import Intervention


class BaselineIntervention(Intervention):
    """Standard public health response without AI interventions.

    Parameters from research design:
    - Care-seeking delay: 3.5 days
    - Clinical diagnosis sensitivity: 75%
    - Surveillance delay: 7 days (manual reporting, lab confirmation)
    - Contact tracing: 40% coverage (Ferretti et al. 2020)
    - No AI tools
    """

    def __init__(self):
        super().__init__()

    def get_name(self) -> str:
        return "baseline"

    def modify_beta(self, beta, day, prevalence):
        """Baseline beta modification: NPI ramp after detection + prevalence fear.

        Even without AI, public health authorities enact NPIs after detection,
        and the general public reduces contacts when prevalence is visible.
        """
        self.check_activation(day, prevalence)

        reduction = 0.0

        # Post-detection: gradual NPI ramp (public health directives)
        if self.outbreak_detected and self.activation_day is not None:
            days_since = day - self.activation_day
            npi_reduction = min(0.15, days_since * 0.01)
            reduction += max(0.0, npi_reduction)

        # Universal prevalence-dependent fear (all agents reduce contacts)
        # Kicks in at 0.5% prevalence — people notice local cases
        # Google Mobility (Goolsbee & Syverson 2021): ~15-20% spontaneous reduction pre-lockdown
        if prevalence > 0.005:
            fear_reduction = min(0.15, (prevalence - 0.005) * 1.5)
            reduction += fear_reduction

        reduction = min(reduction, 0.30)  # Cap total at 30%
        return beta * (1.0 - reduction)

    def get_detection_rate(self, day, prevalence):
        """Baseline detection rate for ODE mode.

        Even without AI, symptomatic agents eventually seek care.
        Base: symptomatic_frac * sensitivity / (incubation + delay)
        = 0.67 * 0.75 / (5.2 + 3.5) ~ 0.058

        After outbreak detection, modest public health response adds +0.02.
        """
        base = 0.058
        if self.outbreak_detected:
            return base + 0.02
        return base

    def get_care_seeking_delay(self, has_ai_access=False):
        return 3.5

    def get_detection_sensitivity(self, has_ai_access=False):
        return 0.75

    def get_contact_tracing_coverage(self, day, prevalence):
        return 0.40  # Ferretti et al. 2020: real COVID tracing ~30-50%

    def get_ai_tool_access_rate(self):
        return 0.0

    def get_surveillance_delay(self):
        return 7.0  # Traditional surveillance: manual reporting, lab confirmation

    def get_forecast_horizon(self):
        return 0.0  # No forecasting

    def get_countermeasure_effect(self, day):
        return 0.0  # No AI-accelerated countermeasures
