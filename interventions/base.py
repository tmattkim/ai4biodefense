"""Abstract base class for intervention scenarios.

All scenarios share a two-stage activation pattern:
1. Record detection_day when prevalence first crosses activation_prevalence
2. Set outbreak_detected = True when day >= detection_day + surveillance_delay

After activation, each scenario responds differently (enhanced detection,
contact tracing, beta reduction) based on their specific capabilities.
"""

from abc import ABC, abstractmethod


class Intervention(ABC):
    """Interface that all intervention scenarios implement.

    Patient-level interventions modify agent-level parameters.
    System-level interventions modify population/compartmental parameters.

    All interventions share outbreak activation state — even baseline
    public health eventually detects an outbreak and responds.
    """

    def __init__(self):
        # Outbreak activation state (shared by all scenarios)
        self.outbreak_detected = False
        self.detection_day = None   # Day prevalence first crossed threshold
        self.activation_day = None  # Day response actually activates
        self.activation_prevalence = 0.001  # 0.1% prevalence threshold

    def check_activation(self, day: float, prevalence: float) -> None:
        """Check and update outbreak detection state.

        Two-stage pattern:
        1. Record detection_day when prevalence crosses threshold
        2. Activate when surveillance delay has elapsed

        Should be called each step by modify_beta() or the engine.
        """
        # Stage 1: record when threshold is first crossed
        if self.detection_day is None and prevalence > self.activation_prevalence:
            self.detection_day = day

        # Stage 2: activate after surveillance delay
        if (not self.outbreak_detected
                and self.detection_day is not None
                and day >= self.detection_day + self.get_surveillance_delay()):
            self.outbreak_detected = True
            self.activation_day = day

    def reset(self):
        """Reset activation state for new replication."""
        self.outbreak_detected = False
        self.detection_day = None
        self.activation_day = None

    @abstractmethod
    def get_name(self) -> str:
        """Return intervention scenario name."""

    @abstractmethod
    def modify_beta(self, beta: float, day: int, prevalence: float) -> float:
        """Modify transmission rate (population-level effect).

        Used in ODE mode and optionally for ABM population-level adjustments.
        """

    @abstractmethod
    def get_detection_rate(self, day: int, prevalence: float) -> float:
        """Return detection/isolation rate delta for ODE Iiso compartment."""

    @abstractmethod
    def get_care_seeking_delay(self, has_ai_access: bool) -> float:
        """Return care-seeking delay in days."""

    @abstractmethod
    def get_detection_sensitivity(self, has_ai_access: bool) -> float:
        """Return diagnostic sensitivity (true positive rate)."""

    @abstractmethod
    def get_contact_tracing_coverage(self, day: int, prevalence: float) -> float:
        """Return fraction of contacts successfully traced."""

    @abstractmethod
    def get_ai_tool_access_rate(self) -> float:
        """Return fraction of population with AI tool access."""

    @abstractmethod
    def get_surveillance_delay(self) -> float:
        """Return days from case occurrence to system-level detection."""

    @abstractmethod
    def get_forecast_horizon(self) -> float:
        """Return days of advance warning from outbreak forecasting."""

    @abstractmethod
    def get_countermeasure_effect(self, day: int) -> float:
        """Return severity/recovery reduction factor from countermeasures.

        Returns float in [0, max_reduction]. 0 = no effect.
        Used by biomanufacturing acceleration (system AI).
        """
