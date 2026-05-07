"""Scenario configuration binding disease config + intervention."""

from dataclasses import dataclass
from config.base import SimulationConfig
from config.interventions.baseline import BaselineIntervention
from interventions.patient_level import PatientLevelAI
from interventions.system_level import SystemLevelAI
from interventions.combined import CombinedAI


@dataclass
class Scenario:
    """Binds a disease configuration with an intervention scenario."""
    name: str
    disease_config: object
    intervention: object
    sim_config: SimulationConfig

    @property
    def display_name(self) -> str:
        disease_name = getattr(self.disease_config, 'display_name', 'Unknown')
        return f"{disease_name} - {self.name}"


def build_scenarios(disease_config, sim_config=None):
    """Build all 4 intervention scenarios for a given disease.

    Returns:
        Dict mapping scenario name to Scenario object.
    """
    if sim_config is None:
        sim_config = SimulationConfig()

    return {
        "baseline": Scenario(
            name="baseline",
            disease_config=disease_config,
            intervention=BaselineIntervention(),
            sim_config=sim_config,
        ),
        "patient_ai": Scenario(
            name="patient_ai",
            disease_config=disease_config,
            intervention=PatientLevelAI(),
            sim_config=sim_config,
        ),
        "system_ai": Scenario(
            name="system_ai",
            disease_config=disease_config,
            intervention=SystemLevelAI(),
            sim_config=sim_config,
        ),
        "combined": Scenario(
            name="combined",
            disease_config=disease_config,
            intervention=CombinedAI(),
            sim_config=sim_config,
        ),
    }
