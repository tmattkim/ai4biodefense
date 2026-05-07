# Aggregated trajectory data

Per (disease, scenario), each CSV contains daily compartment counts aggregated
across 200 replications. Use these to plot mean trajectories with prediction
interval bands (2.5%-97.5%) and inter-quartile bands (25%-75%).

## Schema

Each `<disease>_<scenario>.csv` has columns:

- `day`: simulation day index (integer, 0-indexed)
- For each compartment X in {S, E, I, R, prevalence}:
  - `X_mean`: mean across 200 reps
  - `X_std`: std deviation across 200 reps
  - `X_p2.5`: 2.5th percentile (lower bound of 95% prediction interval)
  - `X_p25`: 25th percentile
  - `X_p50`: 50th percentile (median)
  - `X_p75`: 75th percentile
  - `X_p97.5`: 97.5th percentile (upper bound of 95% prediction interval)
- `n_reps`: number of replications aggregated (200 for all)

S/E/I/R counts are absolute (out of N=10,000). prevalence is fraction (0-1).

## Files

20 files: 5 diseases × 4 scenarios.

Diseases: covid19, influenza, ebola, dengue, measles
Scenarios: baseline, patient_ai, system_ai, combined

## Recommended figure types

- **SEIR curves with PI band**: plot {S,E,I,R}_mean as line, fill_between
  {S,E,I,R}_p2.5 and {S,E,I,R}_p97.5 with low alpha for the band
- **Intervention overlay**: plot I_mean for all 4 scenarios on same axes
- **Box-plot proxy**: at any given day, the p25/p50/p75/p2.5/p97.5 values
  describe the rep distribution
