"""Statistical analysis: paired comparisons, bootstrap CIs, sensitivity analysis."""

import numpy as np
from scipy import stats


def paired_comparison(
    metric_baseline: np.ndarray,
    metric_intervention: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Paired t-test across replications.

    Args:
        metric_baseline: Array of metric values from baseline replications.
        metric_intervention: Array from intervention replications.
        alpha: Significance level.

    Returns:
        Dict with mean_diff, ci_lower, ci_upper, p_value, cohens_d
    """
    diffs = metric_baseline - metric_intervention
    n = len(diffs)

    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))

    if std_diff == 0 or n < 2:
        return {
            "mean_diff": mean_diff,
            "ci_lower": mean_diff,
            "ci_upper": mean_diff,
            "p_value": 0.0 if mean_diff != 0 else 1.0,
            "cohens_d": 0.0,
            "n": n,
        }

    t_stat, p_value = stats.ttest_rel(metric_baseline, metric_intervention)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    se = std_diff / np.sqrt(n)

    # Cohen's d for paired samples
    cohens_d = mean_diff / std_diff

    return {
        "mean_diff": mean_diff,
        "ci_lower": float(mean_diff - t_crit * se),
        "ci_upper": float(mean_diff + t_crit * se),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "t_statistic": float(t_stat),
        "n": n,
    }


def bootstrap_ci(
    data: np.ndarray,
    statistic=np.mean,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    rng: np.random.Generator = None,
) -> tuple:
    """BCa (bias-corrected accelerated) bootstrap confidence interval.

    Args:
        data: 1D array of observations.
        statistic: Function to compute on each bootstrap sample.
        n_bootstrap: Number of bootstrap samples.
        alpha: Significance level.
        rng: Random generator.

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(data)
    point = float(statistic(data))

    # Generate bootstrap samples
    boot_stats = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats[b] = statistic(sample)

    # Percentile method (simpler than BCa, sufficient for most uses)
    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return point, ci_lower, ci_upper


def _eval_model_func(args):
    """Worker function for parallel Sobol evaluation."""
    func, param_dict = args
    return func(param_dict)


def sobol_sensitivity(
    model_func,
    param_ranges: dict,
    n_samples: int = 1024,
    seed: int = 42,
    n_workers: int = 1,
) -> dict:
    """Sobol sensitivity analysis using SALib.

    Args:
        model_func: Callable that takes a dict of parameter values
                    and returns a scalar outcome metric.
        param_ranges: Dict mapping parameter names to (lower, upper) bounds.
                     e.g., {"R0": (2.0, 6.0), "ai_access_rate": (0.1, 0.8)}
        n_samples: Number of Sobol samples (actual runs = n_samples * (2*n_params + 2)).
        seed: Random seed.
        n_workers: Number of parallel workers (1 = sequential).

    Returns:
        Dict with 'S1' (first-order), 'ST' (total-order) indices per parameter,
        and 'S1_conf', 'ST_conf' confidence intervals.
    """
    try:
        from SALib.sample import saltelli
        from SALib.analyze import sobol
    except ImportError:
        raise ImportError("SALib required for Sobol analysis: pip install SALib")

    names = list(param_ranges.keys())
    bounds = [param_ranges[k] for k in names]

    problem = {
        'num_vars': len(names),
        'names': names,
        'bounds': bounds,
    }

    # Generate Sobol samples
    param_values = saltelli.sample(problem, n_samples)

    # Build parameter dicts for all evaluations
    param_dicts = [
        {name: val for name, val in zip(names, row)}
        for row in param_values
    ]

    # Evaluate model (parallel or sequential)
    if n_workers > 1:
        from multiprocessing import Pool
        with Pool(n_workers) as pool:
            Y = np.array(pool.map(
                _eval_model_func,
                [(model_func, pd) for pd in param_dicts],
            ))
    else:
        Y = np.empty(len(param_dicts))
        for i, pd in enumerate(param_dicts):
            Y[i] = model_func(pd)

    # Analyze
    Si = sobol.analyze(problem, Y, print_to_console=False)

    return {
        "parameters": names,
        "S1": Si['S1'].tolist(),
        "S1_conf": Si['S1_conf'].tolist(),
        "ST": Si['ST'].tolist(),
        "ST_conf": Si['ST_conf'].tolist(),
    }


def mixed_effects_regression(
    results_by_disease: dict,
    metric_name: str = "peak_reduction_pct",
) -> dict:
    """Mixed-effects regression for multi-disease intervention comparison.

    Tests whether intervention effectiveness varies significantly
    across disease characteristics.

    Args:
        results_by_disease: Dict mapping disease_name -> dict of
            {scenario_name: list of metric values across replications}
        metric_name: Which metric to analyze.

    Returns:
        Dict with regression results.
    """
    try:
        import statsmodels.formula.api as smf
        import pandas as pd
    except ImportError:
        # Fallback: simple ANOVA
        return _simple_anova(results_by_disease, metric_name)

    rows = []
    for disease, scenarios in results_by_disease.items():
        for scenario, values in scenarios.items():
            if scenario == "baseline":
                continue
            for rep_idx, val in enumerate(values):
                rows.append({
                    "disease": disease,
                    "scenario": scenario,
                    "replication": rep_idx,
                    metric_name: val,
                })

    df = pd.DataFrame(rows)

    if len(df) < 10:
        return {"error": "Insufficient data for mixed-effects regression"}

    # Mixed-effects model: metric ~ scenario + (1|disease)
    model = smf.mixedlm(
        f"{metric_name} ~ C(scenario)",
        data=df,
        groups=df["disease"],
    )
    result = model.fit()

    return {
        "summary": str(result.summary()),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "fixed_effects": {k: float(v) for k, v in result.fe_params.items()},
        "random_effects_variance": float(result.cov_re.iloc[0, 0]),
        "p_values": {k: float(v) for k, v in result.pvalues.items()},
    }


def _simple_anova(results_by_disease, metric_name):
    """Fallback ANOVA when statsmodels is not available."""
    groups = []
    for disease, scenarios in results_by_disease.items():
        for scenario, values in scenarios.items():
            if scenario == "baseline":
                continue
            groups.append(np.array(values))

    if len(groups) < 2:
        return {"error": "Need at least 2 groups for ANOVA"}

    f_stat, p_value = stats.f_oneway(*groups)
    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "n_groups": len(groups),
    }
