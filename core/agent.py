"""Agent data structures and enumerations for the ABM component."""

from dataclasses import dataclass
from enum import IntEnum


class EpiState(IntEnum):
    """Epidemiological compartment states."""
    SUSCEPTIBLE = 0
    EXPOSED = 1
    INFECTED = 2
    RECOVERED = 3


class Severity(IntEnum):
    """Symptom severity levels."""
    ASYMPTOMATIC = 0
    MILD = 1
    MODERATE = 2
    SEVERE = 3
    CRITICAL = 4


class Archetype(IntEnum):
    """Behavioral archetypes grounded in Health Belief Model and social determinants.

    Distribution in 10K population:
    - YOUNG_INVINCIBLE: 12% (1200) — age 18-25, optimism bias
    - WORKING_PARENT: 18% (1800) — age 30-45, competing demands
    - ESSENTIAL_WORKER: 20% (2000) — hourly wage, no sick leave
    - ELDERLY_CAUTIOUS: 12% (1200) — age 65+, high perceived severity
    - HEALTHCARE_INFORMED: 8% (800) — medical/science background
    - DISTRUST_SKEPTIC: 12% (1200) — low institutional trust
    - IMMUNOCOMPROMISED: 8% (800) — underlying conditions
    - COMMUNITY_ORIENTED: 10% (1000) — behavior follows community norms
    """
    YOUNG_INVINCIBLE = 0
    WORKING_PARENT = 1
    ESSENTIAL_WORKER = 2
    ELDERLY_CAUTIOUS = 3
    HEALTHCARE_INFORMED = 4
    DISTRUST_SKEPTIC = 5
    IMMUNOCOMPROMISED = 6
    COMMUNITY_ORIENTED = 7


# Age ranges preferred by each archetype (min_age, max_age)
ARCHETYPE_AGE_RANGES = {
    Archetype.YOUNG_INVINCIBLE: (18, 25),
    Archetype.WORKING_PARENT: (30, 45),
    Archetype.ESSENTIAL_WORKER: (25, 55),
    Archetype.ELDERLY_CAUTIOUS: (65, 100),
    Archetype.HEALTHCARE_INFORMED: (30, 60),
    Archetype.DISTRUST_SKEPTIC: (18, 100),
    Archetype.IMMUNOCOMPROMISED: (18, 100),
    Archetype.COMMUNITY_ORIENTED: (18, 100),
}

# Default health literacy by archetype (0-1 scale)
ARCHETYPE_HEALTH_LITERACY = {
    Archetype.YOUNG_INVINCIBLE: 0.4,
    Archetype.WORKING_PARENT: 0.5,
    Archetype.ESSENTIAL_WORKER: 0.35,
    Archetype.ELDERLY_CAUTIOUS: 0.55,
    Archetype.HEALTHCARE_INFORMED: 0.9,
    Archetype.DISTRUST_SKEPTIC: 0.3,
    Archetype.IMMUNOCOMPROMISED: 0.6,
    Archetype.COMMUNITY_ORIENTED: 0.45,
}

# Default care-seeking propensity by archetype (0-1 scale)
ARCHETYPE_CARE_SEEKING = {
    Archetype.YOUNG_INVINCIBLE: 0.2,
    Archetype.WORKING_PARENT: 0.5,
    Archetype.ESSENTIAL_WORKER: 0.3,
    Archetype.ELDERLY_CAUTIOUS: 0.7,
    Archetype.HEALTHCARE_INFORMED: 0.8,
    Archetype.DISTRUST_SKEPTIC: 0.15,
    Archetype.IMMUNOCOMPROMISED: 0.85,
    Archetype.COMMUNITY_ORIENTED: 0.45,
}


@dataclass
class Agent:
    """Individual agent in the simulation.

    Used for initialization and rich queries. During the ABM hot loop,
    agent states are stored as parallel numpy arrays for performance.
    """
    id: int
    age: int
    age_bin: int                          # Index into config.age_bins (0-4)
    household_id: int

    # Epidemiological state
    epi_state: EpiState = EpiState.SUSCEPTIBLE
    days_in_state: float = 0.0
    severity: Severity = Severity.ASYMPTOMATIC

    # Behavioral attributes
    archetype: Archetype = Archetype.YOUNG_INVINCIBLE
    health_literacy: float = 0.5
    care_seeking_propensity: float = 0.5
    has_ai_tool_access: bool = False

    # Dynamic state
    is_isolated: bool = False
    is_detected: bool = False
    is_hospitalized: bool = False
    risk_perception: float = 0.0
    days_since_symptom_onset: float = -1.0

    # Vaccination (for measles / influenza)
    is_vaccinated: bool = False

    # Transmission tracking (for counterfactual analysis)
    infection_day: float = -1.0
    infector_id: int = -1
