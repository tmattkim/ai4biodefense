"""System-level AI intervention scenario.

Operates on population/compartmental parameters, grounded in two key
biodefense AI applications from the literature:

1. AI-powered genomic surveillance: Protein language models predict viral
   evolution and detect novel variants earlier. This translates to:
   - Faster outbreak detection (surveillance delay 1 day vs 7 baseline)
   - Earlier activation of population-level response measures
   - Enhanced detection/isolation rate through systematic case finding
   - 10-day forecast horizon allows pre-emptive action

2. AI-accelerated biomanufacturing: AI automates and accelerates lab processes,
   countermeasure development, and treatment production. This translates to:
   - Time-dependent reduction in disease severity (as treatments become available)
   - Gradual increase in recovery rate as countermeasures deploy

Both operate at the institutional level with inherent response latency — the
time from pathogen emergence to system-level awareness and action. This
contrasts with patient-level AI which operates at the frontline, directly
interfacing with patients.

References:
    Lytras et al. (2025). "Pathogen Genomic Surveillance and the AI Revolution."
    Green et al. (2024). "A Biomanufacturing Plan to Confront Future Biological Threats."
"""

from interventions.base import Intervention


class SystemLevelAI(Intervention):
    """System-level AI intervention modifying population-level parameters.

    Models two biodefense AI capabilities:
    - Genomic surveillance: earlier outbreak detection, faster case finding
    - Biomanufacturing: accelerated countermeasure development and deployment
    """

    def __init__(
        self,
        forecast_horizon: float = 10.0,
        forecast_activation_prevalence: float = 0.001,
        max_beta_reduction: float = 0.35,
        beta_reduction_rate: float = 0.03,
        # Biomanufacturing parameters
        countermeasure_lead_time: float = 30.0,
        max_severity_reduction: float = 0.30,
        severity_ramp_days: float = 30.0,
    ):
        super().__init__()
        self._forecast_horizon = forecast_horizon
        self.forecast_activation_prevalence = forecast_activation_prevalence
        self.activation_prevalence = forecast_activation_prevalence  # Override base
        self.max_beta_reduction = max_beta_reduction
        self.beta_reduction_rate = beta_reduction_rate

        # Biomanufacturing parameters
        self.countermeasure_lead_time = countermeasure_lead_time
        self.max_severity_reduction = max_severity_reduction
        self.severity_ramp_days = severity_ramp_days

        self._forecast_applied = False

    def get_name(self) -> str:
        return "system_ai"

    def modify_beta(self, beta, day, prevalence):
        """Genomic surveillance triggers earlier population-level response.

        Two-stage activation with surveillance delay and forecast horizon:
        1. Threshold crossing: records detection_day
        2. Activation: after surveillance_delay (1 day for AI genomic surveillance)
        3. Forecast horizon: shifts activation_day backward for pre-emptive action

        After activation, gradual beta reduction from population-level measures
        (targeted interventions, resource deployment, public health directives).
        """
        self.check_activation(day, prevalence)

        # Apply forecast horizon on first activation (shift activation_day backward)
        if self.outbreak_detected and self.activation_day is not None and not self._forecast_applied:
            lead = max(0.0, self._forecast_horizon - self.get_surveillance_delay())
            self.activation_day = self.activation_day - lead
            self._forecast_applied = True

        reduction = 0.0

        if self.outbreak_detected and self.activation_day is not None:
            days_since = day - self.activation_day
            # Institutional response: targeted quarantines, venue closures,
            # ring vaccination, travel advisories — enabled by genomic surveillance
            reduction = min(
                self.max_beta_reduction,
                self.beta_reduction_rate * max(0.0, days_since),
            )

        # Universal prevalence-dependent fear (weakened, same as baseline)
        # Google Mobility (Goolsbee & Syverson 2021): ~15-20% spontaneous reduction
        if prevalence > 0.005:
            fear = min(0.15, (prevalence - 0.005) * 1.5)
            reduction += fear

        reduction = min(reduction, 0.50)  # Cap at 50%
        return beta * (1.0 - reduction)

    def get_detection_rate(self, day, prevalence):
        """Combined self-detection + AI genomic surveillance detection rate.

        Agents still self-detect (like baseline) plus AI surveillance systems
        enable systematic case finding. Higher delta = faster I -> Iiso flow.
        """
        self_detection = 0.058
        if self.outbreak_detected:
            surveillance = 0.12  # Enhanced AI-powered surveillance
        else:
            surveillance = 0.02  # Pre-activation baseline surveillance
        return self_detection + surveillance

    def get_care_seeking_delay(self, has_ai_access=False):
        return 3.5  # No patient-level effect

    def get_detection_sensitivity(self, has_ai_access=False):
        return 0.75  # No patient-level effect

    def get_contact_tracing_coverage(self, day, prevalence):
        if self.outbreak_detected:
            return 0.70  # Enhanced from 0.40 via genomic data
        return 0.40

    def get_ai_tool_access_rate(self):
        return 0.0  # No patient-level AI tools

    def get_surveillance_delay(self):
        return 1.0  # AI genomic surveillance: rapid but needs sample + sequencing

    def get_forecast_horizon(self):
        return self._forecast_horizon

    def get_countermeasure_effect(self, day):
        """Return severity reduction factor from AI-accelerated biomanufacturing.

        After outbreak activation + lead_time, countermeasures (treatments,
        therapeutics) begin deploying, gradually reducing effective disease
        severity. This models how AI accelerates the drug development pipeline.

        Returns:
            Float in [0, max_severity_reduction], representing the fraction
            by which disease severity is reduced (0 = no effect, 0.3 = 30%).
        """
        if not self.outbreak_detected or self.activation_day is None:
            return 0.0

        days_since = day - self.activation_day
        if days_since < self.countermeasure_lead_time:
            return 0.0

        deploy_days = days_since - self.countermeasure_lead_time
        ramp = min(1.0, deploy_days / self.severity_ramp_days)
        return self.max_severity_reduction * ramp

    def reset(self):
        """Reset state for new replication."""
        super().reset()
        self._forecast_applied = False
