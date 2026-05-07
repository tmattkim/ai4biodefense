"""Patient-level AI (Medical AI) intervention scenario.

Operates as frontline biodefense through direct patient interaction,
grounded in the emerging Medical AI (MAI) literature:

1. AI symptom checker: 40% population access (ramping to 60% during outbreak),
   reduces care-seeking delay 3.5 -> 1.5 days. Nudges reluctant agents toward
   seeking care (+0.3 care-seeking propensity boost). Based on Semigran et al.
   (2015) audit of symptom checker accuracy.

2. Clinical diagnostic AI: +20% detection sensitivity. AI-powered diagnostics
   in radiology and pathology now match or exceed human performance
   (Najjar 2023, McGenity et al. 2024).

3. AI-assisted contact monitoring: -50% secondary case detection delay.
   Wearable data analysis and automated contact notification accelerate
   the contact tracing pipeline (Moritz et al. 2025).

4. Voluntary behavior change: Symptomatic AI users reduce their own
   transmission by 50% (masking, distancing) before formal isolation.
   COVID-era data shows self-aware symptomatic individuals reduced
   contacts by 40-60% (Jarvis et al. 2020).

Unlike system-level AI which operates at the institutional level with
inherent response latency, patient-level AI operates at the frontline —
directly interfacing with patients at the point where pathogens spread.
Effects are mediated through individual agent behavior, modulated by
each agent's archetype (care-seeking propensity, health literacy).
"""

import numpy as np
from interventions.base import Intervention


class PatientLevelAI(Intervention):
    """Patient-level Medical AI modifying agent-level parameters.

    Models frontline biodefense through direct patient interface.
    Effects are heterogeneous across behavioral archetypes.
    """

    def __init__(
        self,
        ai_access_rate: float = 0.40,
        max_ai_access_rate: float = 0.60,
        reduced_delay: float = 1.5,
        sensitivity_boost: float = 0.20,
        contact_monitoring_reduction: float = 0.50,
        voluntary_reduction_factor: float = 0.50,
    ):
        super().__init__()
        self.ai_access_rate = ai_access_rate
        self.max_ai_access_rate = max_ai_access_rate
        self.reduced_delay = reduced_delay
        self.sensitivity_boost = sensitivity_boost
        self.contact_monitoring_reduction = contact_monitoring_reduction
        self.voluntary_reduction_factor = voluntary_reduction_factor

    def get_name(self) -> str:
        return "patient_ai"

    def modify_beta(self, beta, day, prevalence):
        """Behavioral beta reduction from voluntary behavior + AI-informed proactive response.

        Three components:
        1. Symptomatic AI users voluntarily reduce transmission
        2. AI-informed proactive contact reduction (key patient AI differentiator)
        3. Universal prevalence-dependent fear (same weak formula as baseline)

        ODE approximation combines all three effects.
        """
        self.check_activation(day, prevalence)

        ai_rate = self.get_dynamic_ai_access_rate(day, prevalence)

        # 1. Voluntary behavior change (symptomatic AI users)
        symp_frac = 0.67
        voluntary_effect = ai_rate * symp_frac * self.voluntary_reduction_factor

        # 2. AI-informed proactive contact reduction (the key patient AI differentiator)
        # AI users have real-time dashboards/alerts, respond at 0.1% prevalence
        # (10x earlier than general fear at 0.5%)
        # Jarvis et al. 2020: informed individuals reduced contacts 40-60%
        ai_proactive = 0.0
        if prevalence > 0.001:
            ai_proactive = ai_rate * min(0.35, (prevalence - 0.001) * 4.0)

        # 3. Universal prevalence-dependent fear (weakened, same as baseline)
        # Google Mobility (Goolsbee & Syverson 2021): ~15-20% spontaneous reduction
        fear = min(0.15, (prevalence - 0.005) * 1.5) if prevalence > 0.005 else 0.0

        total_reduction = voluntary_effect + ai_proactive + fear
        total_reduction = min(total_reduction, 0.60)  # Cap at 60%
        return beta * (1.0 - total_reduction)

    def get_detection_rate(self, day, prevalence):
        """Effective population-level detection rate for ODE mode.

        Approximates the ABM self-detection mechanism:
        - AI users: higher sensitivity, shorter delay
        - Non-AI users: baseline detection
        - After outbreak detection: +0.02 from public health response
        """
        symp_frac = 0.67
        ai_rate = self.get_dynamic_ai_access_rate(day, prevalence)
        delta_ai = symp_frac * (self.sensitivity_boost + 0.75) / (5.2 + self.reduced_delay)
        delta_base = symp_frac * 0.75 / (5.2 + 3.5)
        rate = ai_rate * delta_ai + (1 - ai_rate) * delta_base
        if self.outbreak_detected:
            rate += 0.02
        return rate

    def get_care_seeking_delay(self, has_ai_access=False):
        if has_ai_access:
            return self.reduced_delay  # 1.5 days
        return 3.5  # Baseline for non-AI users

    def get_detection_sensitivity(self, has_ai_access=False):
        base = 0.75
        if has_ai_access:
            return min(base + self.sensitivity_boost, 1.0)  # 0.95
        return base

    def get_contact_tracing_coverage(self, day, prevalence):
        return 0.40  # Same as baseline; improvement is in speed (delay factor)

    def get_ai_tool_access_rate(self):
        return self.ai_access_rate

    def get_dynamic_ai_access_rate(self, day, prevalence):
        """AI access ramps from base to max during outbreak.

        Models surge in symptom checker app downloads during an epidemic.
        Sigmoid growth centered at 1% prevalence.
        """
        if prevalence <= 0:
            return self.ai_access_rate

        threshold = 0.01
        steepness = 200.0
        sigmoid = 1.0 / (1.0 + np.exp(-steepness * (prevalence - threshold)))
        return self.ai_access_rate + (self.max_ai_access_rate - self.ai_access_rate) * sigmoid

    def get_surveillance_delay(self):
        return 5.0  # Faster than baseline: more early detections surface the signal

    def get_forecast_horizon(self):
        return 0.0  # No system-level forecasting

    def get_countermeasure_effect(self, day):
        return 0.0  # No AI-accelerated countermeasures

    def get_contact_detection_delay_factor(self):
        """Factor applied to secondary case detection delay.

        AI-assisted monitoring reduces delay by 50%.
        """
        return 1.0 - self.contact_monitoring_reduction  # 0.5

    def get_voluntary_reduction_factor(self):
        """Transmission reduction for symptomatic AI users in ABM mode."""
        return self.voluntary_reduction_factor

    def reset(self):
        """Reset state for new replication."""
        super().reset()
