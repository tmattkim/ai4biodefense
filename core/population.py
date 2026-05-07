"""Population initialization: create agents with demographics, households, archetypes."""

import numpy as np
from core.agent import (
    Agent, EpiState, Archetype,
    ARCHETYPE_AGE_RANGES, ARCHETYPE_HEALTH_LITERACY, ARCHETYPE_CARE_SEEKING,
)
from config.base import SimulationConfig


def _assign_age_bin(age: int, age_bins: tuple) -> int:
    """Map an age to its bin index."""
    for i, (lo, hi) in enumerate(age_bins):
        if lo <= age <= hi:
            return i
    return len(age_bins) - 1


def _generate_ages(n: int, config: SimulationConfig, rng: np.random.Generator) -> np.ndarray:
    """Sample ages from US Census distribution within each age bin."""
    ages = np.empty(n, dtype=int)
    idx = 0
    for i, (lo, hi) in enumerate(config.age_bins):
        count = int(round(config.age_distribution[i] * n))
        if i == len(config.age_bins) - 1:
            count = n - idx  # Ensure we fill exactly n
        count = min(count, n - idx)
        ages[idx:idx + count] = rng.integers(lo, hi + 1, size=count)
        idx += count
    rng.shuffle(ages)
    return ages


def _create_households(n: int, mean_size: float, rng: np.random.Generator) -> np.ndarray:
    """Assign agents to households with Poisson-distributed sizes."""
    household_ids = np.empty(n, dtype=int)
    hh_id = 0
    idx = 0
    while idx < n:
        size = max(1, rng.poisson(mean_size))
        size = min(size, n - idx)
        household_ids[idx:idx + size] = hh_id
        hh_id += 1
        idx += size
    return household_ids


def _assign_archetypes(
    agents: list, config: SimulationConfig, rng: np.random.Generator
) -> None:
    """Assign archetypes respecting age constraints and target distribution."""
    n = len(agents)
    archetype_counts = [int(round(p * n)) for p in config.archetype_distribution]
    # Fix rounding to ensure sum == n
    diff = n - sum(archetype_counts)
    archetype_counts[2] += diff  # Adjust Essential Worker (largest group)

    # Build pool of archetype slots
    archetype_pool = []
    for arch_idx, count in enumerate(archetype_counts):
        archetype_pool.extend([Archetype(arch_idx)] * count)
    rng.shuffle(archetype_pool)

    # Assign: prefer matching age range, but don't block if no match
    assigned = [False] * n
    agent_indices = list(range(n))
    rng.shuffle(agent_indices)

    # First pass: assign to agents within preferred age range
    for arch in Archetype:
        lo, hi = ARCHETYPE_AGE_RANGES[arch]
        target_count = archetype_counts[arch.value]
        filled = 0
        for i in agent_indices:
            if assigned[i]:
                continue
            if lo <= agents[i].age <= hi and filled < target_count:
                agents[i].archetype = arch
                agents[i].health_literacy = ARCHETYPE_HEALTH_LITERACY[arch] + rng.normal(0, 0.1)
                agents[i].health_literacy = np.clip(agents[i].health_literacy, 0, 1)
                agents[i].care_seeking_propensity = ARCHETYPE_CARE_SEEKING[arch] + rng.normal(0, 0.1)
                agents[i].care_seeking_propensity = np.clip(agents[i].care_seeking_propensity, 0, 1)
                assigned[i] = True
                filled += 1
        # Second pass: fill remaining from unassigned agents regardless of age
        if filled < target_count:
            for i in agent_indices:
                if assigned[i]:
                    continue
                if filled >= target_count:
                    break
                agents[i].archetype = arch
                agents[i].health_literacy = ARCHETYPE_HEALTH_LITERACY[arch] + rng.normal(0, 0.1)
                agents[i].health_literacy = np.clip(agents[i].health_literacy, 0, 1)
                agents[i].care_seeking_propensity = ARCHETYPE_CARE_SEEKING[arch] + rng.normal(0, 0.1)
                agents[i].care_seeking_propensity = np.clip(agents[i].care_seeking_propensity, 0, 1)
                assigned[i] = True
                filled += 1


def _assign_vaccination(agents: list, disease_config, rng: np.random.Generator) -> None:
    """Assign vaccination status based on disease-specific coverage.

    Vaccination coverage varies by archetype:
    - Distrust/Skeptic: 50% of base coverage (vaccine hesitancy)
    - Healthcare Informed: 120% of base coverage (capped at 1.0)
    - Immunocompromised: 110% of base coverage
    - Others: base coverage

    For measles (MMR), clustered non-vaccination is important —
    household members share vaccination status to model pockets of
    unvaccinated communities (e.g., 2025 Texas outbreak).
    """
    coverage = getattr(disease_config, 'vaccine_coverage', 0.0)
    if coverage <= 0:
        return

    # Cluster vaccination by household for diseases with clustered non-vaccination
    is_clustered = disease_config.name in ('measles',)

    if is_clustered:
        # Assign at household level: each household is vaccinated or not
        households = {}
        for a in agents:
            households.setdefault(a.household_id, []).append(a)

        for hh_id, members in households.items():
            # Archetype-adjusted coverage for household (use first member's archetype)
            rep_arch = members[0].archetype
            adj_coverage = _archetype_vaccine_adjustment(coverage, rep_arch)
            hh_vaccinated = rng.random() < adj_coverage
            for a in members:
                a.is_vaccinated = hh_vaccinated
    else:
        # Individual-level vaccination
        for a in agents:
            adj_coverage = _archetype_vaccine_adjustment(coverage, a.archetype)
            a.is_vaccinated = rng.random() < adj_coverage


def _archetype_vaccine_adjustment(base_coverage: float, archetype) -> float:
    """Adjust vaccine coverage by behavioral archetype."""
    if archetype == Archetype.DISTRUST_SKEPTIC:
        return base_coverage * 0.50
    elif archetype == Archetype.HEALTHCARE_INFORMED:
        return min(1.0, base_coverage * 1.20)
    elif archetype == Archetype.IMMUNOCOMPROMISED:
        return min(1.0, base_coverage * 1.10)
    elif archetype == Archetype.YOUNG_INVINCIBLE:
        return base_coverage * 0.80
    return base_coverage


def create_population(config: SimulationConfig, disease_config, rng: np.random.Generator) -> list:
    """Create the full agent population.

    Args:
        config: Global simulation configuration.
        disease_config: Disease-specific configuration (for initial seeding).
        rng: NumPy random generator for reproducibility.

    Returns:
        List of Agent objects.
    """
    n = config.population_size
    ages = _generate_ages(n, config, rng)
    household_ids = _create_households(n, config.mean_household_size, rng)

    agents = []
    for i in range(n):
        age_bin = _assign_age_bin(int(ages[i]), config.age_bins)
        agents.append(Agent(
            id=i,
            age=int(ages[i]),
            age_bin=age_bin,
            household_id=int(household_ids[i]),
        ))

    # Assign archetypes with demographic-aware distribution
    _assign_archetypes(agents, config, rng)

    # Assign vaccination based on disease-specific coverage
    vaccine_coverage = getattr(disease_config, 'vaccine_coverage', 0.0)
    if vaccine_coverage > 0:
        _assign_vaccination(agents, disease_config, rng)

    # Seed initial infections (only from unvaccinated or vaccine-breakthrough)
    susceptible_indices = list(range(n))
    rng.shuffle(susceptible_indices)

    # Scale initial conditions (disease configs specify per 10,000 population)
    scale = n / 10_000
    n_exposed = max(1, round(getattr(disease_config, 'initial_exposed', 0) * scale))
    n_infected = max(1, round(getattr(disease_config, 'initial_infected', 10) * scale))

    for i in range(min(n_exposed, n)):
        agents[susceptible_indices[i]].epi_state = EpiState.EXPOSED
        agents[susceptible_indices[i]].days_in_state = rng.uniform(
            0, disease_config.incubation_period
        )

    for i in range(n_exposed, min(n_exposed + n_infected, n)):
        agents[susceptible_indices[i]].epi_state = EpiState.INFECTED
        agents[susceptible_indices[i]].days_in_state = rng.uniform(
            0, disease_config.infectious_period * 0.3
        )
        agents[susceptible_indices[i]].infection_day = 0.0

    return agents


def assign_ai_access(agents: list, access_rate: float, rng: np.random.Generator) -> None:
    """Assign AI tool access correlated with health literacy.

    Higher health literacy -> higher probability of having AI tool access.
    Distrust/Skeptic archetype has reduced access probability.
    """
    for agent in agents:
        base_prob = access_rate
        # Modulate by health literacy
        prob = base_prob * (0.5 + agent.health_literacy)
        # Skeptics are less likely to adopt AI tools
        if agent.archetype == Archetype.DISTRUST_SKEPTIC:
            prob *= 0.3
        prob = np.clip(prob, 0, 1)
        agent.has_ai_tool_access = rng.random() < prob
