"""Combined AI intervention scenario.

Both patient-level and system-level interventions with interaction effects:
- Patient-level AI data streams feed into surveillance (better forecasting)
- System-level targeting directs AI tools to high-risk populations
- 15% synergy factor on complementary mechanisms
"""

from interventions.base import Intervention
from interventions.patient_level import PatientLevelAI
from interventions.system_level import SystemLevelAI


class CombinedAI(Intervention):
    """Combined patient + system AI with synergy effects."""

    def __init__(
        self,
        ai_access_rate: float = 0.40,
        max_ai_access_rate: float = 0.60,
        reduced_delay: float = 1.5,
        sensitivity_boost: float = 0.20,
        contact_monitoring_reduction: float = 0.50,
        voluntary_reduction_factor: float = 0.50,
        forecast_horizon: float = 10.0,
        synergy_factor: float = 1.15,
    ):
        super().__init__()
        self.patient = PatientLevelAI(
            ai_access_rate, max_ai_access_rate, reduced_delay,
            sensitivity_boost, contact_monitoring_reduction,
            voluntary_reduction_factor,
        )
        self.system = SystemLevelAI(
            forecast_horizon=forecast_horizon,
        )
        self.synergy_factor = synergy_factor

    def get_name(self) -> str:
        return "combined"

    def modify_beta(self, beta, day, prevalence):
        # Ensure both sub-interventions update their activation state
        self.patient.check_activation(day, prevalence)
        self.system.check_activation(day, prevalence)

        # Apply forecast horizon (same logic as SystemLevelAI.modify_beta)
        if (self.system.outbreak_detected and self.system.activation_day is not None
                and not self.system._forecast_applied):
            lead = max(0.0, self.system._forecast_horizon - self.system.get_surveillance_delay())
            self.system.activation_day = self.system.activation_day - lead
            self.system._forecast_applied = True

        # 1. Patient-specific: voluntary behavior change (unique to patient AI)
        ai_rate = self.patient.get_dynamic_ai_access_rate(day, prevalence)
        voluntary = ai_rate * 0.67 * self.patient.voluntary_reduction_factor

        # 2. Patient-specific: AI-informed proactive contact reduction
        ai_proactive = 0.0
        if prevalence > 0.001:
            ai_proactive = ai_rate * min(0.35, (prevalence - 0.001) * 4.0)

        # 3. System-specific: institutional response (unique to system AI)
        institutional = 0.0
        if self.system.outbreak_detected and self.system.activation_day is not None:
            days_since = day - self.system.activation_day
            institutional = min(
                self.system.max_beta_reduction,
                self.system.beta_reduction_rate * max(0.0, days_since),
            )

        # 4. Shared: universal prevalence-dependent fear (applied ONCE, weakened)
        fear = min(0.15, (prevalence - 0.005) * 1.5) if prevalence > 0.005 else 0.0

        # Combine: independent effects multiply (no double-counting)
        combined = 1.0 - (1.0 - voluntary) * (1.0 - ai_proactive) * (1.0 - institutional) * (1.0 - fear)

        # Synergy bonus
        total_reduction = combined * self.synergy_factor
        total_reduction = min(total_reduction, 0.75)  # Cap at 75%

        # Update own activation state from system component
        self.outbreak_detected = self.system.outbreak_detected
        self.detection_day = self.system.detection_day
        self.activation_day = self.system.activation_day

        return beta * (1.0 - total_reduction)

    def get_detection_rate(self, day, prevalence):
        """Combined detection: shared base + unique patient AI + unique system AI.

        Base self-detection (0.058) is shared — agents only self-detect once.
        Patient AI adds faster/better detection for AI users.
        System AI adds independent surveillance case-finding.
        """
        # Shared: base symptomatic self-detection (all scenarios include this)
        base_self_detection = 0.058

        # Patient-unique: AI access improves detection speed/sensitivity
        patient_rate = self.patient.get_detection_rate(day, prevalence)
        patient_unique = patient_rate - base_self_detection
        # Remove the outbreak boost from patient (avoid double-counting with system)
        if self.patient.outbreak_detected:
            patient_unique -= 0.02

        # System-unique: AI surveillance case-finding (independent of self-detection)
        system_rate = self.system.get_detection_rate(day, prevalence)
        system_unique = system_rate - base_self_detection

        # Outbreak public health boost (applied once)
        outbreak_boost = 0.02 if self.outbreak_detected else 0.0

        combined = (base_self_detection + max(0.0, patient_unique)
                    + max(0.0, system_unique) + outbreak_boost)
        combined *= self.synergy_factor
        return min(combined, 0.5)

    def get_care_seeking_delay(self, has_ai_access=False):
        return self.patient.get_care_seeking_delay(has_ai_access)

    def get_detection_sensitivity(self, has_ai_access=False):
        return self.patient.get_detection_sensitivity(has_ai_access)

    def get_contact_tracing_coverage(self, day, prevalence):
        if self.system.outbreak_detected:
            return 0.80  # Combined AI-enhanced tracing
        return 0.40

    def get_ai_tool_access_rate(self):
        base_rate = self.patient.get_ai_tool_access_rate()
        if self.system.outbreak_detected:
            return min(base_rate * self.synergy_factor, 0.60)
        return base_rate

    def get_surveillance_delay(self):
        return 0.5  # AI surveillance + patient-detected cases feed directly into system

    def get_forecast_horizon(self):
        return self.system.get_forecast_horizon() * self.synergy_factor

    def get_countermeasure_effect(self, day):
        return self.system.get_countermeasure_effect(day)

    def get_contact_detection_delay_factor(self):
        return self.patient.get_contact_detection_delay_factor()

    def get_voluntary_reduction_factor(self):
        return self.patient.get_voluntary_reduction_factor()

    def get_dynamic_ai_access_rate(self, day, prevalence):
        """Delegate to patient component for dynamic AI adoption."""
        return self.patient.get_dynamic_ai_access_rate(day, prevalence)

    def reset(self):
        """Reset state for new replication."""
        super().reset()
        self.patient.reset()
        self.system.reset()
