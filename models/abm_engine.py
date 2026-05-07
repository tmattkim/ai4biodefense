"""Vectorized Agent-Based Model simulation engine.

Agent states are stored as parallel numpy arrays (Structure of Arrays)
for performance. The Agent dataclass is used only for initialization.
"""

import numpy as np
from core.agent import EpiState, Severity, Archetype
from core.contact_network import get_combined_adjacency


class ABMEngine:
    """Core ABM simulation engine with vectorized state transitions.

    Stores agent attributes as numpy arrays for fast computation.
    Uses stochastic SEIR transitions: P = 1 - exp(-rate * dt).
    """

    def __init__(self, agents: list, networks: dict, disease_config, dt: float = 1.0,
                 rng: np.random.Generator = None):
        """Initialize from Agent list and contact networks.

        Args:
            agents: List of Agent objects (used to populate arrays).
            networks: Dict of NetworkX graphs from build_contact_network().
            disease_config: Disease configuration object.
            dt: Time step in days.
            rng: NumPy random generator.
        """
        self.n = len(agents)
        self.dt = dt
        self.disease_config = disease_config
        self.rng = rng or np.random.default_rng()

        # ---- Vectorize agent attributes ----
        self.states = np.array([a.epi_state for a in agents], dtype=np.int8)
        self.ages = np.array([a.age for a in agents], dtype=np.int16)
        self.age_bins = np.array([a.age_bin for a in agents], dtype=np.int8)
        self.household_ids = np.array([a.household_id for a in agents], dtype=np.int32)
        self.archetypes = np.array([a.archetype for a in agents], dtype=np.int8)
        self.health_literacy = np.array([a.health_literacy for a in agents], dtype=np.float32)
        self.care_seeking = np.array([a.care_seeking_propensity for a in agents], dtype=np.float32)
        self.has_ai_access = np.array([a.has_ai_tool_access for a in agents], dtype=bool)
        self.is_vaccinated = np.array([a.is_vaccinated for a in agents], dtype=bool)

        # Dynamic state arrays
        self.days_in_state = np.array([a.days_in_state for a in agents], dtype=np.float32)
        self.severity = np.array([a.severity for a in agents], dtype=np.int8)
        self.isolated = np.array([a.is_isolated for a in agents], dtype=bool)
        self.detected = np.array([a.is_detected for a in agents], dtype=bool)
        self.hospitalized = np.array([a.is_hospitalized for a in agents], dtype=bool)
        self.days_since_symptoms = np.array([a.days_since_symptom_onset for a in agents], dtype=np.float32)
        self.risk_perception = np.array([a.risk_perception for a in agents], dtype=np.float32)

        # Transmission tracking
        self.infection_day = np.full(self.n, -1.0, dtype=np.float32)
        self.infector_id = np.full(self.n, -1, dtype=np.int32)

        # Seed initial infection days
        infected_mask = self.states == EpiState.INFECTED
        self.infection_day[infected_mask] = 0.0

        # Pre-generate incubation and infectious period durations (gamma-distributed)
        self.incubation_duration = self.rng.gamma(
            shape=4.0,
            scale=disease_config.incubation_period / 4.0,
            size=self.n,
        ).astype(np.float32)
        self.infectious_duration = self.rng.gamma(
            shape=4.0,
            scale=disease_config.infectious_period / 4.0,
            size=self.n,
        ).astype(np.float32)

        # Build adjacency structure from networks
        self._build_adjacency(agents, networks)

        # Tag household edges in adjacency (for isolated household transmission)
        source_ids = np.repeat(np.arange(self.n), np.diff(self.adj_offsets))
        self.adj_is_household = (
            self.household_ids[source_ids] == self.household_ids[self.adj_neighbors]
        )

        # Assign initial symptom severity for infected agents
        self._assign_severity(infected_mask)

        # Superspreading: individual transmissibility multiplier
        # Drawn from Gamma(k, 1/k) so E[individual_beta] = 1.0
        self.individual_beta = np.ones(self.n, dtype=np.float32)
        k_disp = getattr(disease_config, 'dispersion_k', None)
        if k_disp is not None:
            n_init = int(infected_mask.sum())
            if n_init > 0:
                self.individual_beta[infected_mask] = self.rng.gamma(
                    k_disp, 1.0 / k_disp, size=n_init
                ).astype(np.float32)

        # Essential worker flag (derived from archetype)
        self.essential_worker = (self.archetypes == Archetype.ESSENTIAL_WORKER)

        # Compliance fatigue: track when isolation started
        self.isolation_start_day = np.full(self.n, -1.0, dtype=np.float32)

        # Perceived prevalence: real-time for AI users, lagged for non-AI
        self.perceived_prevalence = np.zeros(self.n, dtype=np.float32)
        self.reported_prevalence = 0.0
        self.reported_prevalence_history = []

        # Care-seeking tracking: True once agent has attempted care-seeking
        self.care_sought = np.zeros(self.n, dtype=bool)

        # Cumulative counters
        self.cumulative_infections = int(infected_mask.sum())
        self.cumulative_hospitalizations = 0
        self.cumulative_deaths = 0
        self.cumulative_detections = 0
        self.current_day = 0.0

        # Pending isolation queue for contact tracing: list of (agent_idx, isolation_day)
        self._pending_isolation = []

        # History tracking
        self.transmission_log = []

    def _build_adjacency(self, agents, networks):
        """Convert NetworkX graphs to fast adjacency structure.

        Stores as CSR-like arrays for vectorized neighbor lookup.
        """
        adjacency = get_combined_adjacency(networks, agents)

        # Build CSR-like structure
        neighbors_list = []
        weights_list = []
        offsets = np.zeros(self.n + 1, dtype=np.int32)

        for i in range(self.n):
            nbrs = adjacency.get(i, [])
            for nbr_id, weight in nbrs:
                neighbors_list.append(nbr_id)
                weights_list.append(weight)
            offsets[i + 1] = len(neighbors_list)

        self.adj_neighbors = np.array(neighbors_list, dtype=np.int32)
        self.adj_weights = np.array(weights_list, dtype=np.float32)
        self.adj_offsets = offsets

    def _assign_severity(self, mask):
        """Assign symptom severity for newly infected agents."""
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return

        for idx in indices:
            age_bin = self.age_bins[idx]
            symp_frac = self.disease_config.symptomatic_fraction_by_age[age_bin]
            if self.rng.random() > symp_frac:
                self.severity[idx] = Severity.ASYMPTOMATIC
            else:
                # Lognormal severity: 1=mild, 2=moderate, 3=severe, 4=critical
                sev_val = self.rng.lognormal(
                    self.disease_config.severity_log_mean,
                    self.disease_config.severity_log_std,
                )
                if sev_val < 1.5:
                    self.severity[idx] = Severity.MILD
                elif sev_val < 3.0:
                    self.severity[idx] = Severity.MODERATE
                elif sev_val < 5.0:
                    self.severity[idx] = Severity.SEVERE
                else:
                    self.severity[idx] = Severity.CRITICAL

    def step(self, day: float, isolation_decisions: np.ndarray = None,
             intervention=None) -> dict:
        """Execute one time step of the ABM.

        Args:
            day: Current simulation day.
            isolation_decisions: Optional boolean array of isolation decisions
                                from the behavioral engine.
            intervention: Optional Intervention object for self-detection
                         parameters (care-seeking delay, detection sensitivity).

        Returns:
            Dict with step results:
                new_infections: count of new E transitions
                new_infectious: count of new E->I transitions
                new_recoveries: count of new I->R transitions
                state_counts: array [S, E, I, R] total counts
                state_counts_by_age: shape (4, n_age_bins) array
                transmission_events: list of (infector, infectee) tuples
        """
        self.current_day = day

        # 0. Update dynamic AI access if intervention supports it
        self._update_dynamic_ai_access(day, intervention)

        # 0.5. Update perceived prevalence (used for behavioral response + contact reduction)
        self._update_perceived_prevalence(day, intervention)

        # 1. Apply isolation decisions from behavioral engine
        if isolation_decisions is not None:
            newly_isolated = isolation_decisions & (~self.isolated)
            self.isolated |= isolation_decisions
            self.isolation_start_day[np.where(newly_isolated)[0]] = day

        # 1.5. Process contact-traced isolations due this step
        ct_isolations = self._process_pending_isolations(day, intervention)

        # 1.6. Compliance fatigue: isolated agents may break isolation
        self._process_compliance_fatigue()

        # 2. Self-detection: symptomatic agents seek care and may self-isolate
        new_detections = self._self_detection_step(intervention)

        # 3. Compute force of infection and new infections (S -> E)
        new_exposed_mask, transmission_events = self._transmission_step(intervention)

        # 4. E -> I transitions
        new_infected_mask = self._exposed_to_infected()

        # 5. I -> R transitions (including death)
        new_recovered_mask, new_deaths = self._infected_to_recovered(intervention)

        # 6. Update days in state
        self.days_in_state += self.dt

        # 7. Update symptom onset tracking
        infected_symptomatic = (
            (self.states == EpiState.INFECTED) &
            (self.severity > Severity.ASYMPTOMATIC) &
            (self.days_since_symptoms < 0)
        )
        self.days_since_symptoms[infected_symptomatic] = 0.0
        symptomatic = self.days_since_symptoms >= 0
        self.days_since_symptoms[symptomatic] += self.dt

        # 8. Hospitalization check for severe/critical
        new_hosp = self._check_hospitalization(new_infected_mask)

        # 9. Update risk perception based on prevalence
        prevalence = np.sum(self.states == EpiState.INFECTED) / self.n
        self._update_risk_perception(prevalence)

        # Track results
        n_new_exposed = int(new_exposed_mask.sum())
        n_new_infected = int(new_infected_mask.sum())
        n_new_recovered = int(new_recovered_mask.sum())

        self.cumulative_infections += n_new_exposed
        self.transmission_log.extend(transmission_events)

        state_counts = np.array([
            np.sum(self.states == EpiState.SUSCEPTIBLE),
            np.sum(self.states == EpiState.EXPOSED),
            np.sum(self.states == EpiState.INFECTED),
            np.sum(self.states == EpiState.RECOVERED),
        ])

        state_counts_by_age = np.zeros((4, 5), dtype=int)
        for s in range(4):
            for ab in range(5):
                state_counts_by_age[s, ab] = np.sum(
                    (self.states == s) & (self.age_bins == ab)
                )

        return {
            "day": day,
            "new_infections": n_new_exposed,
            "new_infectious": n_new_infected,
            "new_recoveries": n_new_recovered,
            "new_deaths": new_deaths,
            "new_hospitalizations": new_hosp,
            "new_detections": new_detections + ct_isolations,
            "state_counts": state_counts,
            "state_counts_by_age": state_counts_by_age,
            "prevalence": float(prevalence),
            "isolation_rate": float(self.isolated.mean()),
            "transmission_events": transmission_events,
        }

    def _self_detection_step(self, intervention):
        """Symptomatic agents self-detect and isolate based on care-seeking behavior.

        Models the pathway: symptoms -> care-seeking delay -> diagnostic test -> isolation.
        Each agent's behavior depends on their archetype through:
        - care_seeking_propensity: modulates delay and probability of seeking care
        - health_literacy: modulates diagnostic accuracy

        Patient-level AI reduces the base care-seeking delay (3.5 -> 1.5 days) and
        increases detection sensitivity (75% -> 95%) for agents with AI tool access.
        AI also nudges reluctant agents toward care-seeking (+0.3 propensity boost).

        Args:
            intervention: Intervention object with get_care_seeking_delay() and
                         get_detection_sensitivity() methods.

        Returns:
            Number of new detections this step.
        """
        if intervention is None:
            return 0

        # Find symptomatic infected agents who haven't yet sought care
        candidates = (
            (self.states == EpiState.INFECTED) &
            (~self.isolated) &
            (~self.care_sought) &
            (self.severity > Severity.ASYMPTOMATIC) &
            (self.days_since_symptoms >= 0)
        )

        indices = np.where(candidates)[0]
        new_detections = 0

        for idx in indices:
            has_ai = bool(self.has_ai_access[idx])
            cs = float(self.care_seeking[idx])  # 0.15 (Skeptic) to 0.85 (Immunocompromised)
            hl = float(self.health_literacy[idx])

            # Archetype modulates care-seeking delay:
            # Higher propensity -> shorter delay (proactive agents seek care sooner)
            # Scale: 0.5x delay (very proactive) to 1.5x delay (very reluctant)
            base_delay = intervention.get_care_seeking_delay(has_ai)
            delay_multiplier = 0.5 + (1.0 - cs)
            effective_delay = base_delay * delay_multiplier

            if self.days_since_symptoms[idx] >= effective_delay:
                self.care_sought[idx] = True

                # Archetype modulates probability of actually seeking care
                # care_seeking_propensity = base probability of following through
                # AI symptom checker nudges reluctant agents (+0.3)
                seek_prob = cs
                if has_ai:
                    seek_prob = min(1.0, cs + 0.3)

                # Severity also increases care-seeking urgency
                sev = int(self.severity[idx])
                if sev >= Severity.SEVERE:
                    seek_prob = min(1.0, seek_prob + 0.3)
                elif sev >= Severity.MODERATE:
                    seek_prob = min(1.0, seek_prob + 0.1)

                # Prevalence-dependent care-seeking boost (all agents)
                awareness = float(self.perceived_prevalence[idx])
                seek_prob = min(1.0, seek_prob + awareness * 2.0)

                if self.rng.random() >= seek_prob:
                    continue  # Agent decides not to seek care

                # Agent seeks care — detection depends on diagnostic sensitivity
                # Health literacy slightly improves detection (better symptom description)
                base_sensitivity = intervention.get_detection_sensitivity(has_ai)
                effective_sensitivity = base_sensitivity * (0.8 + 0.2 * hl)

                if self.rng.random() < effective_sensitivity:
                    self.detected[idx] = True
                    new_detections += 1
                    self.cumulative_detections += 1

                    # Essential workers: 50% chance cannot isolate (work obligations)
                    if self.essential_worker[idx] and self.rng.random() > 0.5:
                        # Detected but cannot isolate — still trigger tracing
                        self._contact_tracing_step(idx, self.current_day, intervention)
                        if has_ai:
                            hh_mask = self.household_ids == self.household_ids[idx]
                            self.care_seeking[hh_mask] = np.minimum(
                                self.care_seeking[hh_mask] + 0.2, 1.0
                            )
                        continue

                    self.isolated[idx] = True  # Home isolation
                    self.isolation_start_day[idx] = self.current_day

                    # Trigger contact tracing for newly-detected case
                    self._contact_tracing_step(idx, self.current_day, intervention)

                    # Household spillover: boost care-seeking for household members
                    if has_ai:
                        hh_mask = self.household_ids == self.household_ids[idx]
                        self.care_seeking[hh_mask] = np.minimum(
                            self.care_seeking[hh_mask] + 0.2, 1.0
                        )

        return new_detections

    def _transmission_step(self, intervention=None):
        """Compute stochastic S -> E transitions based on contact network.

        Transmission sources:
        1. Non-isolated infected agents (standard transmission)
        2. Isolated infected agents to household contacts only (epsilon=0.1)
        3. Pre-symptomatic EXPOSED agents in last 2 days of incubation

        Modifiers per source agent:
        - individual_beta: superspreading multiplier (Gamma-distributed)
        - Voluntary reduction: symptomatic AI users reduce transmission
        - Contact reduction: non-household weights reduced by perceived prevalence

        Returns:
            (new_exposed_mask, transmission_events)
        """
        beta = self.disease_config.beta

        # Seasonal forcing (influenza): beta oscillates with season
        seasonal_amp = getattr(self.disease_config, 'seasonal_amplitude', 0.0)
        if seasonal_amp > 0:
            peak_day = getattr(self.disease_config, 'seasonal_peak_day', 0)
            beta *= 1.0 + seasonal_amp * np.cos(2.0 * np.pi * (self.current_day - peak_day) / 365.0)

        # Apply population-level beta modification (NPIs, fear response)
        # This mirrors the ODE modify_beta and ensures consistent behavior
        if intervention is not None:
            prevalence = float(np.sum(self.states == EpiState.INFECTED)) / self.n
            beta = intervention.modify_beta(beta, self.current_day, prevalence)

        susceptible = np.where(self.states == EpiState.SUSCEPTIBLE)[0]

        # Three sets of potential transmitters
        infected_set = set(np.where(
            (self.states == EpiState.INFECTED) & (~self.isolated)
        )[0])
        isolated_infected_set = set(np.where(
            (self.states == EpiState.INFECTED) & self.isolated
        )[0])

        # Pre-symptomatic: EXPOSED agents in last 2 days of incubation
        presymp_frac = getattr(self.disease_config, 'presymptomatic_fraction', 0.0)
        presymp_set = set()
        if presymp_frac > 0:
            exposed_mask = self.states == EpiState.EXPOSED
            time_to_transition = self.incubation_duration - self.days_in_state
            presymp_mask = exposed_mask & (time_to_transition <= 2.0) & (time_to_transition > 0)
            presymp_set = set(np.where(presymp_mask)[0])

        new_exposed = np.zeros(self.n, dtype=bool)
        transmission_events = []

        # Note: voluntary behavior reduction for symptomatic AI users is already
        # captured in modify_beta() as a population-level average. Do NOT apply
        # vol_factor again here — that would double-count the effect.

        # Vaccination effect
        vaccine_eff = getattr(self.disease_config, 'vaccine_efficacy_infection', 0.0)

        # Isolated household transmission leakage rate
        epsilon = 0.1

        # Disease-specific contact weight modifiers
        hh_tx_mult = getattr(self.disease_config, 'household_transmission_multiplier', 1.0)
        comm_tx_mult = getattr(self.disease_config, 'community_transmission_multiplier', 1.0)
        contact_scaling = getattr(self.disease_config, 'contact_scaling_factor', 1.0)

        for s_idx in susceptible:
            start = self.adj_offsets[s_idx]
            end = self.adj_offsets[s_idx + 1]

            lambda_i = 0.0
            potential_infectors = []

            # Prevalence-dependent contact reduction for this agent
            # Modest supplementary effect on top of modify_beta's universal fear response
            # Captures individual-level avoidance heterogeneity (AI users respond earlier)
            awareness = float(self.perceived_prevalence[s_idx])
            contact_reduction = min(0.25, awareness * 2.0)  # Cap at 25%
            community_factor = max(0.75, 1.0 - contact_reduction)

            for k in range(start, end):
                nbr = self.adj_neighbors[k]
                is_hh = bool(self.adj_is_household[k])
                w = float(self.adj_weights[k])

                # Disease-specific contact weight modifiers
                if is_hh:
                    w *= hh_tx_mult
                else:
                    w *= comm_tx_mult * contact_scaling

                # Contact reduction: only for non-household edges
                if not is_hh:
                    w *= community_factor

                contribution = 0.0

                if nbr in infected_set:
                    # Standard non-isolated infected
                    effective_w = w * self.individual_beta[nbr]
                    contribution = beta * effective_w

                elif nbr in isolated_infected_set and is_hh:
                    # Isolated infected → household contacts only at epsilon rate
                    contribution = beta * epsilon * w * self.individual_beta[nbr]

                elif nbr in presymp_set:
                    # Pre-symptomatic EXPOSED agent
                    if self.isolated[nbr]:
                        # Contact-traced and isolated: household only at epsilon
                        if is_hh:
                            contribution = beta * presymp_frac * epsilon * w * self.individual_beta[nbr]
                    else:
                        # w already has community_factor applied for non-household edges (line 464)
                        effective_w = w * self.individual_beta[nbr]
                        contribution = beta * presymp_frac * effective_w

                if contribution > 0:
                    lambda_i += contribution
                    potential_infectors.append(nbr)

            if lambda_i <= 0:
                continue

            # Stochastic infection probability
            p_infect = 1.0 - np.exp(-lambda_i * self.dt)

            # Vaccination reduces susceptibility
            if self.is_vaccinated[s_idx] and vaccine_eff > 0:
                p_infect *= (1.0 - vaccine_eff)

            if self.rng.random() < p_infect:
                new_exposed[s_idx] = True
                self.states[s_idx] = EpiState.EXPOSED
                self.days_in_state[s_idx] = 0.0
                self.infection_day[s_idx] = self.current_day

                # Draw individual_beta at S->E (so pre-symptomatic agents have it)
                k_disp = getattr(self.disease_config, 'dispersion_k', None)
                if k_disp is not None:
                    self.individual_beta[s_idx] = float(
                        self.rng.gamma(k_disp, 1.0 / k_disp)
                    )

                # Record infector (choose proportional to weight)
                if potential_infectors:
                    infector = self.rng.choice(potential_infectors)
                    self.infector_id[s_idx] = infector
                    transmission_events.append((int(infector), int(s_idx)))

        return new_exposed, transmission_events

    def _exposed_to_infected(self):
        """E -> I transitions when incubation period ends."""
        exposed = (self.states == EpiState.EXPOSED)
        transition = exposed & (self.days_in_state >= self.incubation_duration)

        indices = np.where(transition)[0]
        self.states[indices] = EpiState.INFECTED
        self.days_in_state[indices] = 0.0
        self._assign_severity(transition)

        return transition

    def _infected_to_recovered(self, intervention=None):
        """I -> R transitions when infectious period ends. Includes death.

        Countermeasure effect (from system AI biomanufacturing) reduces IFR
        and accelerates recovery as treatments become available.
        """
        # Countermeasure effect: reduces IFR and accelerates recovery
        cm_effect = 0.0
        if intervention is not None:
            cm_effect = float(intervention.get_countermeasure_effect(int(self.current_day)))

        infected = (self.states == EpiState.INFECTED)

        # Effective recovery threshold: shorter if countermeasures deployed
        if cm_effect > 0:
            effective_threshold = self.infectious_duration / (1.0 + cm_effect)
        else:
            effective_threshold = self.infectious_duration

        transition = infected & (self.days_in_state >= effective_threshold)

        indices = np.where(transition)[0]
        deaths = 0

        # Burial transmission probability (Ebola: unsafe burial practices)
        burial_prob = getattr(self.disease_config, 'burial_transmission_probability', 0.0)

        for idx in indices:
            age_bin = self.age_bins[idx]
            ifr = self.disease_config.ifr_by_age[age_bin]
            effective_ifr = ifr * (1.0 - cm_effect)
            died = self.rng.random() < effective_ifr
            if died:
                deaths += 1
                self.cumulative_deaths += 1

                # Burial transmission: dead agent can infect household contacts
                if burial_prob > 0:
                    hh_mask = (self.household_ids == self.household_ids[idx])
                    hh_susceptible = np.where(
                        hh_mask & (self.states == EpiState.SUSCEPTIBLE)
                    )[0]
                    for s_idx in hh_susceptible:
                        if self.rng.random() < burial_prob:
                            self.states[s_idx] = EpiState.EXPOSED
                            self.days_in_state[s_idx] = 0.0
                            self.infection_day[s_idx] = self.current_day
                            self.infector_id[s_idx] = idx
                            self.cumulative_infections += 1
                            k_disp = getattr(self.disease_config, 'dispersion_k', None)
                            if k_disp is not None:
                                self.individual_beta[s_idx] = float(
                                    self.rng.gamma(k_disp, 1.0 / k_disp)
                                )

            self.states[idx] = EpiState.RECOVERED
            self.days_in_state[idx] = 0.0
            self.isolated[idx] = False
            self.detected[idx] = False
            self.hospitalized[idx] = False
            self.care_sought[idx] = False
            self.severity[idx] = Severity.ASYMPTOMATIC
            self.days_since_symptoms[idx] = -1.0
            self.individual_beta[idx] = 1.0
            self.isolation_start_day[idx] = -1.0

        return transition, deaths

    def _process_pending_isolations(self, day, intervention):
        """Isolate contact-traced agents whose delay has elapsed.

        Returns number of new isolations from contact tracing this step.
        """
        if not self._pending_isolation or intervention is None:
            return 0

        new_ct_isolations = 0

        remaining = []
        for agent_idx, isolation_day in self._pending_isolation:
            if day < isolation_day:
                remaining.append((agent_idx, isolation_day))
                continue
            # Due: isolate if agent is infected/exposed and not already isolated
            if (self.states[agent_idx] in (EpiState.INFECTED, EpiState.EXPOSED)
                    and not self.isolated[agent_idx]):
                self.detected[agent_idx] = True
                self.cumulative_detections += 1
                # Essential workers: 50% chance cannot isolate
                if self.essential_worker[agent_idx] and self.rng.random() > 0.5:
                    new_ct_isolations += 1
                    continue
                self.isolated[agent_idx] = True
                self.isolation_start_day[agent_idx] = day
                new_ct_isolations += 1

        self._pending_isolation = remaining
        return new_ct_isolations

    def _contact_tracing_step(self, detected_idx, day, intervention):
        """Schedule isolation of network contacts of a newly detected agent.

        Realistic contact tracing: interviews the detected case to find their
        network neighbors (adjacency contacts), then schedules isolation for
        infected/exposed contacts with probability = tracing_coverage.
        """
        prevalence = float(np.sum(self.states == EpiState.INFECTED)) / self.n
        coverage = intervention.get_contact_tracing_coverage(int(day), prevalence)

        # Base delay: 2 days, reduced by AI contact monitoring
        base_delay = 2.0
        delay_factor = 1.0
        if hasattr(intervention, 'get_contact_detection_delay_factor'):
            delay_factor = intervention.get_contact_detection_delay_factor()
        elif hasattr(intervention, 'patient'):
            delay_factor = intervention.patient.get_contact_detection_delay_factor()
        effective_delay = base_delay * delay_factor

        # Collect network contacts (adjacency neighbors of detected case)
        start = self.adj_offsets[detected_idx]
        end = self.adj_offsets[detected_idx + 1]
        neighbors = self.adj_neighbors[start:end]

        # Schedule isolation for infected/exposed contacts with probability = coverage
        for contact_idx in neighbors:
            contact_idx = int(contact_idx)
            if contact_idx == detected_idx:
                continue
            # Only trace contacts who are currently infected or exposed (pre-symptomatic)
            state = self.states[contact_idx]
            if state != EpiState.INFECTED and state != EpiState.EXPOSED:
                continue
            if self.isolated[contact_idx]:
                continue
            if self.rng.random() < coverage:
                self._pending_isolation.append((contact_idx, day + effective_delay))

    def _update_dynamic_ai_access(self, day, intervention):
        """Periodically reassign AI access as adoption grows during outbreak.

        Only applies to interventions with dynamic AI access (patient AI, combined).
        Updates every 5 days to avoid overhead.
        """
        if intervention is None:
            return
        if not hasattr(intervention, 'get_dynamic_ai_access_rate'):
            return
        # Only update every 5 days
        if day > 0 and int(day / self.dt) % int(5 / self.dt) != 0:
            return

        prevalence = float(np.sum(self.states == EpiState.INFECTED)) / self.n
        target_rate = intervention.get_dynamic_ai_access_rate(day, prevalence)
        current_rate = float(self.has_ai_access.mean())

        if target_rate > current_rate + 0.01:
            # Need to grant access to more agents
            no_access = np.where(~self.has_ai_access)[0]
            n_new = int((target_rate - current_rate) * self.n)
            if n_new > 0 and len(no_access) > 0:
                n_new = min(n_new, len(no_access))
                new_access = self.rng.choice(no_access, size=n_new, replace=False)
                self.has_ai_access[new_access] = True

    def _check_hospitalization(self, new_infected_mask):
        """Check if newly infectious agents require hospitalization."""
        indices = np.where(new_infected_mask)[0]
        new_hosp = 0
        for idx in indices:
            age_bin = self.age_bins[idx]
            hosp_rate = self.disease_config.hosp_rate_by_age[age_bin]
            if self.severity[idx] >= Severity.SEVERE and self.rng.random() < hosp_rate:
                self.hospitalized[idx] = True
                self.isolated[idx] = True  # Hospitalized = isolated
                self.isolation_start_day[idx] = self.current_day
                new_hosp += 1
                self.cumulative_hospitalizations += 1
        return new_hosp

    def _update_risk_perception(self, prevalence):
        """Update agent risk perception based on global and local prevalence.

        risk_perception = alpha * global_prevalence + (1-alpha) * local_prevalence
        """
        alpha = 0.6  # Weight on global vs local info

        # Local prevalence: fraction of household members infected
        for hh_id in np.unique(self.household_ids):
            hh_mask = self.household_ids == hh_id
            hh_infected = np.sum((self.states[hh_mask] == EpiState.INFECTED))
            hh_size = np.sum(hh_mask)
            local_prev = hh_infected / max(hh_size, 1)

            self.risk_perception[hh_mask] = (
                alpha * prevalence + (1 - alpha) * local_prev
            )

    def _update_perceived_prevalence(self, day, intervention):
        """Update each agent's perceived prevalence based on information access.

        AI users get real-time prevalence; non-AI users get lagged prevalence
        based on surveillance delay (traditional reporting and lab confirmation).
        """
        real_prevalence = float(np.sum(self.states == EpiState.INFECTED)) / self.n

        # Store history for lagged lookback
        self.reported_prevalence_history.append((day, real_prevalence))
        # Trim to 30 calendar days of history
        cutoff = day - 30.0
        self.reported_prevalence_history = [
            (d, p) for d, p in self.reported_prevalence_history if d >= cutoff
        ]

        # Determine surveillance delay
        surv_delay = 7.0  # default
        if intervention is not None and hasattr(intervention, 'get_surveillance_delay'):
            surv_delay = intervention.get_surveillance_delay()

        # Find lagged prevalence
        lagged_prevalence = 0.0
        target_day = day - surv_delay
        for hist_day, hist_prev in reversed(self.reported_prevalence_history):
            if hist_day <= target_day:
                lagged_prevalence = hist_prev
                break

        self.reported_prevalence = lagged_prevalence

        # AI users: real-time; non-AI users: lagged
        self.perceived_prevalence[self.has_ai_access] = real_prevalence
        self.perceived_prevalence[~self.has_ai_access] = lagged_prevalence

    def _process_compliance_fatigue(self):
        """Isolated agents may break isolation over time.

        Fatigue increases after a base threshold (7 days default).
        Archetype-dependent: Young Invincibles and Essential Workers fatigue faster;
        Elderly Cautious and Immunocompromised are more compliant.
        """
        # Exclude hospitalized agents — they can't leave the hospital via fatigue
        isolated_mask = self.isolated & (self.isolation_start_day >= 0) & (~self.hospitalized)
        indices = np.where(isolated_mask)[0]
        if len(indices) == 0:
            return

        base_threshold = 7.0
        fatigue_rate = 0.05  # per day past threshold
        max_break_prob = 0.3

        for idx in indices:
            days_isolated = self.current_day - self.isolation_start_day[idx]

            # Archetype multiplier for threshold
            arch = int(self.archetypes[idx])
            if arch == Archetype.ELDERLY_CAUTIOUS or arch == Archetype.IMMUNOCOMPROMISED:
                threshold = base_threshold * 2.0
            elif arch == Archetype.YOUNG_INVINCIBLE or arch == Archetype.ESSENTIAL_WORKER:
                threshold = base_threshold * 0.5
            else:
                threshold = base_threshold

            break_prob = 0.0

            if days_isolated > threshold:
                break_prob = min(max_break_prob,
                                (days_isolated - threshold) * fatigue_rate)

            # Essential workers: additional independent break probability
            if self.essential_worker[idx]:
                break_prob = min(0.5, break_prob + 0.10)

            if break_prob > 0 and self.rng.random() < break_prob:
                self.isolated[idx] = False
                self.isolation_start_day[idx] = -1.0

    def get_local_infection_pressure(self) -> np.ndarray:
        """Compute fraction of household members infected for each agent.

        Returns:
            Float array of shape (n,) with values in [0, 1].
        """
        pressure = np.zeros(self.n, dtype=np.float32)
        for hh_id in np.unique(self.household_ids):
            hh_mask = self.household_ids == hh_id
            hh_size = int(np.sum(hh_mask))
            if hh_size <= 1:
                continue
            hh_infected = int(np.sum(self.states[hh_mask] == EpiState.INFECTED))
            # Exclude the agent itself from the denominator for their own pressure
            hh_indices = np.where(hh_mask)[0]
            for idx in hh_indices:
                others_infected = hh_infected
                if self.states[idx] == EpiState.INFECTED:
                    others_infected -= 1
                pressure[idx] = others_infected / (hh_size - 1)
        return pressure

    def get_state_counts_by_age(self) -> np.ndarray:
        """Return 5 x n_age_bins array of [S, E, I, R, Iiso] counts."""
        n_age = 5
        counts = np.zeros((5, n_age), dtype=int)
        for s in range(4):
            for ab in range(n_age):
                counts[s, ab] = np.sum(
                    (self.states == s) & (self.age_bins == ab)
                )
        # Iiso: isolated infected
        for ab in range(n_age):
            counts[4, ab] = np.sum(
                (self.states == EpiState.INFECTED) &
                self.isolated &
                (self.age_bins == ab)
            )
        return counts

    def sync_from_ode(self, state_proportions_by_age: np.ndarray, rng: np.random.Generator):
        """Reassign agent states from ODE compartment proportions.

        Used when switching from ODE mode back to ABM mode.
        Preserves household clustering: infected agents are preferentially
        placed in households that already had cases.

        Args:
            state_proportions_by_age: shape (5, n_age_bins) array of
                [S, E, I, R, Iiso] proportions per age bin.
            rng: Random generator.
        """
        for ab in range(5):
            ab_mask = self.age_bins == ab
            ab_indices = np.where(ab_mask)[0]
            n_ab = len(ab_indices)

            if n_ab == 0:
                continue

            # Target counts from ODE proportions
            proportions = state_proportions_by_age[:, ab]
            # Normalize to ensure they sum to population in this age bin
            proportions = proportions / proportions.sum()

            counts = np.round(proportions * n_ab).astype(int)
            # Fix rounding
            diff = n_ab - counts.sum()
            counts[0] += diff  # Adjust susceptible

            # Shuffle and assign
            rng.shuffle(ab_indices)
            idx = 0
            state_map = [EpiState.SUSCEPTIBLE, EpiState.EXPOSED,
                         EpiState.INFECTED, EpiState.RECOVERED, EpiState.INFECTED]
            k_disp = getattr(self.disease_config, 'dispersion_k', None)
            for s, count in enumerate(counts):
                for _ in range(count):
                    if idx >= n_ab:
                        break
                    agent_idx = ab_indices[idx]
                    self.states[agent_idx] = state_map[s]
                    # Reset all dynamic flags to clean state
                    self.isolation_start_day[agent_idx] = -1.0
                    self.individual_beta[agent_idx] = 1.0
                    self.detected[agent_idx] = False
                    self.hospitalized[agent_idx] = False
                    self.care_sought[agent_idx] = False
                    self.severity[agent_idx] = Severity.ASYMPTOMATIC
                    self.days_since_symptoms[agent_idx] = -1.0
                    if s == 4:  # Iiso
                        self.isolated[agent_idx] = True
                        self.isolation_start_day[agent_idx] = self.current_day
                    else:
                        self.isolated[agent_idx] = False
                    # Draw individual_beta for infected/exposed agents
                    if s in (1, 2, 4) and k_disp is not None:
                        self.individual_beta[agent_idx] = float(
                            rng.gamma(k_disp, 1.0 / k_disp)
                        )
                    self.days_in_state[agent_idx] = 0.0
                    idx += 1
