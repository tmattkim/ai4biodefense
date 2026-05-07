# Preventing the Proliferation of Bioweapons: Artificial Intelligence for Biodefense

A hybrid SEIR + agent-based simulation framework for evaluating AI-driven biodefense interventions across pathogen classes. Companion code for the undergraduate honors thesis *Preventing the Proliferation of Bioweapons: Artificial Intelligence for Biodefense*.

## Abstract

The convergence of artificial intelligence (AI) and biotechnology has begun to transform biological weapons from a state-monopoly threat constrained by tacit expertise into a capability accessible to non-state actors. Existing biodefense doctrine emphasizes upstream prevention, but such measures address only whether an attack can be initiated, not what occurs once initial defenses fail. Of the three strategic postures available against bioweapons — deterrence by punishment, dissuasion, and denial — only deterrence by denial remains viable in the AI era. Deterrence by punishment is undermined by structural attribution failures, and deterrence by dissuasion is eroded as AI lowers the capability barrier. The same technological convergence that has widened the offensive frontier also enables a robust defensive response that deterrence by denial requires. AI can be deployed as a biodefense capability, operating at two distinct levels: patient-level AI interventions operate through individual agent decisions and probabilistic care-seeking behaviors, and system-level AI interventions modify transmission parameters and detection capabilities at the population level.

This thesis investigates the following research question: *how effective are patient-level AI interventions compared to system-level AI interventions in responding to disease outbreaks?* To address this question, the framework implements four intervention regimes (no AI, patient-level AI, system-level AI, and integrated deployment) across five pathogens spanning the relevant epidemiological parameter space.

Three findings emerge:
1. Both AI interventions reduce attack rates and severity outcomes substantially relative to baseline, with integrated deployment dominating either tier in isolation.
2. The two interventions exhibit complementary specialization — patient-level AI acting most effectively on transmission and system-level AI acting most effectively on severity.
3. Against fast-incubation pathogens — the threat profile most relevant to engineered bioweapons — only integrated deployment produces a meaningful effect, with the synergy between the two interventions far exceeding the sum of their individual contributions.

These findings indicate that deterrence by denial in the AI era requires a biodefense architecture that integrates both layers. Neither alone is sufficient against the most strategically concerning threat profiles.

---

## What this framework models

**Two engines, switching by prevalence.** When prevalence is low (<20%) the simulation runs an explicit agent-based model: 10,000 individual agents on a contact network, each with a state (S/E/I/R), age, behavioral archetype, household, and individual transmission rate (Gamma-distributed for superspreading). When prevalence rises above the threshold, the simulation switches to an age-stratified compartmental SEIR ODE for efficiency, and switches back when prevalence falls. State synchronization preserves S+E+I+R+Iiso = N at every transition.

**Eight behavioral archetypes** (Young Invincible, Working Parent, Essential Worker, Elderly Cautious, Healthcare Informed, Distrust Skeptic, Immunocompromised, Community Oriented), each with distinct care-seeking propensity, health literacy, and isolation tendency.

**Nine realism features**:
1. Prevalence-dependent behavior for all agents (lagged for non-AI users)
2. Superspreading via Gamma-distributed individual transmission rate
3. Isolated household transmission (epsilon = 0.1 leakage)
4. Essential-worker isolation exemption (50% probability cannot fully isolate)
5. Compliance fatigue (archetype-dependent)
6. Pre-symptomatic transmission
7. Baseline NPI ramp after first detection
8. Prevalence-dependent contact-weight reduction
9. `modify_beta` applied in both ABM and ODE modes

**Five pathogens**:

| Pathogen | R0 | Notes |
|---|---|---|
| COVID-19 (Delta) | 4.0 | Superspreading (k = 0.10), pre-symptomatic transmission |
| Influenza (H3N2) | 1.7 | Seasonal forcing |
| Ebola (2014 W. Africa anchor) | 1.8 | Household clustering, burial transmission |
| Dengue (host-host SEIR proxy) | 3.0 | Cross-class robustness probe |
| Measles (Texas 2025 anchor) | 15.0 | Household-clustered vaccination, 85% MMR coverage |

**Four intervention scenarios**:
- `baseline` — no AI; standard public health
- `patient_ai` — patient-level AI (symptom checkers, AI-augmented diagnostics) acting on individual agent decisions and care-seeking
- `system_ai` — system-level AI (genomic surveillance, AI-accelerated biomanufacturing) acting on transmission parameters and detection
- `combined` — both tiers, with synergistic interaction

---

## Headline result

Across **5 pathogens × 4 intervention scenarios × 200 replications = 4,000 simulations**, both AI tiers significantly outperform baseline public health for every pathogen. Combined deployment dominates either alone.

For COVID-19 (Delta), the combined intervention reduces:
- **Cumulative attack rate from 99.97% → 54.36%** (−45.6 pp; *p* < 1e-254, Cohen's *d* = +18.5)
- **Peak active infections from 4,102 → 665** (−83.8%; *d* = +30.6)
- **Hospital-days from 8,910 → 3,871** (−56.6%; *d* = +28.4)

Results are validated with a multi-tier validation suite (mass conservation, ABM-ODE agreement, hybrid switching invariants, intervention dose-response, counterfactual identification, replication adequacy, etc.). See `output/validation/` for all artifacts.

---

## Repository layout

```
.
├── README.md                          ← you are here
├── LICENSE                            ← MIT
├── requirements.txt                   ← Python dependencies
├── .gitignore
│
├── core/                              ← Agent dataclass, population, contact network
├── config/                            ← SimulationConfig, disease configs, baseline intervention
├── models/                            ← ABM engine, hybrid engine, SEIR ODE solver
├── interventions/                     ← Patient AI, System AI, Combined AI classes
├── experiments/                       ← Counterfactual analyzer, runner, parallel orchestration
├── analysis/                          ← Metrics, statistics, validation suite
├── scripts/                           ← Entry points (run_disease.py, run_validation.py)
│
└── output/
    ├── aggregated_trajectories/       ← Mean + 95% PI band per (disease × scenario), 20 CSVs
    ├── validation/                    ← Validation layer artifacts (JSON + CSV)
    └── <disease>/                     ← Per-disease outputs (5 directories)
        ├── all_replications.csv       ← 800-row flat metrics file (200 reps × 4 scenarios)
        └── summary.json               ← Per-scenario stats + pairwise comparisons
```

Per-replication trajectory CSVs (4,000 files, ~120 MB) are not included in the release; the aggregated trajectories provide mean and 95% prediction-interval bands sufficient for figure regeneration.

---

## Setup

Requires Python 3.13+.

```bash
git clone <repo-url>
cd preventing-bioweapons-ai

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running experiments

### Quick smoke test (5 replications, ~1 minute)
```bash
python scripts/run_disease.py covid19 -n 5 --workers 4
```

### Full thesis result (~3 hours on 8 cores)
```bash
python scripts/run_disease.py all -n 200 --workers 8
```
This runs 5 diseases × 4 scenarios × 200 replications = 4,000 simulations. The runner is **resumability-safe**: if interrupted, just rerun and it picks up from completed replications.

Outputs land in `output/<disease>/<scenario>/rep_NNN.json` and `trajectory_NNN.csv`. After completion, summary JSON and aggregated CSV are written automatically.

---

## Validation

```bash
# Tier 1 (default): core layers + L4'/L5/L7/L8
python scripts/run_validation.py --workers 8

# Add Tier 2 (engine math + dose-response): + L2b/L3b/L6
python scripts/run_validation.py --tier 2 --workers 8

# Add Tier 3 (replication adequacy): + L10
python scripts/run_validation.py --tier 3 --workers 8

# Core only (no extended layers)
python scripts/run_validation.py --tier 0
```

Validation layers:

**Core (always run):**
- **L1** — ODE analytical: mass conservation + final-size verification
- **L2** — ABM-ODE agreement under realism (prediction-interval coverage + normalized MAE)
- **L3** — Hybrid switching TOST equivalence (peak + attack rate)
- **L4** — Per-disease face validity (Kermack-McKendrick final-size equation)

**Tier 1:**
- **L4'** — Mode-log telemetry across saved replications (verifies ABM-time fraction)
- **L5** — Realism feature ablation suite (7 conditions)
- **L7** — Counterfactual identification (population/network identity, reset, replay)
- **L8** — Effective R0 measurement (next-generation matrix from infection trees)

**Tier 2:**
- **L2b** — Stripped-ABM convergence to ODE (engine math diagnostic)
- **L3b** — Hybrid switching invariants (mass conservation, monotone-R, switch smoothness)
- **L6** — Intervention dose-response (Patient AI access × System AI lead time)

**Tier 3:**
- **L10** — Replication adequacy / CI plateau

All artifacts written to `output/validation/`. The reference run results are included in this release.

---

## Key files in `output/`

### `output/aggregated_trajectories/<disease>_<scenario>.csv`

Mean + 95% prediction interval band trajectories aggregated across 200 replications. Columns include `day, S_mean, S_std, S_p2.5, S_p25, S_p50, S_p75, S_p97.5, ...` (same for E, I, R, prevalence), `n_reps`. See the README in that folder for the full schema.

These files are sufficient to plot any SEIR curve, intervention overlay, or prediction-interval band figure.

### `output/<disease>/all_replications.csv`

Flat per-replication metrics: 200 reps × 4 scenarios = 800 rows per disease. Columns: `cumulative_attack_rate, peak_incidence, peak_timing, epidemic_duration, hospital_days, total_infections, R_eff_trajectory`. Use for box plots, distribution analyses, and bootstrap CI plots.

### `output/<disease>/summary.json`

Pre-computed per-scenario means/std/95% CI for every metric, plus all 18 pairwise comparisons (paired Wilcoxon p-values, Cohen's d) and synergy scores.

### `output/validation/`

Validation layer artifacts (JSON + CSV) from the reference run. Contains the ODE trajectory, ABM-ODE convergence curves with prediction interval band, hybrid switching diagnostics, mode-log telemetry across all 4,000 production replications, realism feature ablation grid, intervention dose-response curves, counterfactual identification hashes, and effective R0 measurements.

---

## Reproducibility

- **Deterministic seeding**: all RNG flows from `SimulationConfig.seed + replication_index` and `hashlib.sha256(scenario_name)` for cross-scenario subseeding. The same seed produces bit-identical results across runs and machines.
- **Counterfactual identity verified** (validation Layer 7): same seed → identical population hash, identical contact network hash, identical metrics on intervention reset and analyzer-instance replay.
- **Frozen dataclasses**: `SimulationConfig` and all disease configs are frozen, so parameter values can't drift across replications.
- **Resumable**: each replication writes its own JSON to disk; rerunning the same command picks up missing replications without recomputation.

---

## Limitations & caveats

1. **Effective R0 differs from nominal R0** by 40-291% across diseases due to network heterogeneity, superspreading, and pre-symptomatic transmission. Both numbers are reported in `output/validation/L8_effective_R0.json`.
2. **Influenza face-validity gap**: simulated attack rate ≈ 100% vs Kermack-McKendrick analytic 69% at R0 = 1.7. Cause: seasonal forcing pushes effective R0 above nominal over the 365-day horizon.
3. **Dengue is modeled as host-host SEIR** (proxy for a vector-borne pathogen). Vector dynamics, seasonal mosquito populations, and vector-control interventions are out of scope. Dengue results serve as a robustness probe for cross-class generalization, not a vector-borne policy simulation.
4. **Measles is anchored to the 2025 Texas outbreak** for vaccine_coverage (0.85) and hospitalization rate anchors. The simulated population is generic US Census, not Texas-specific.
5. **Ebola attack rates are highly stochastic** at N = 10,000 — many runs produce no sustained outbreak. Effect sizes (Cohen's d ≈ 0.5–0.85) are smaller than for other diseases as a result.

---

## Citation

If you use this framework, please cite:

> Kim, M. (2026). *Preventing the Proliferation of Bioweapons: Artificial Intelligence for Biodefense.* Undergraduate Honors Thesis.

A BibTeX entry will be added when the thesis is publicly archived.

---

## License

MIT. See `LICENSE`.
