import numpy as np
from scipy import stats

def bootstrap_confidence_interval(results, metric_key='ensemble_risk', n_bootstrap=1000, confidence=0.95):
    """
    Compute bootstrap confidence intervals for risk scores.
    
    Returns dict with lower, upper bounds per sample.
    """
    scores = np.array([r[metric_key] for r in results])
    n_samples = len(scores)
    
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_scores.append(scores[indices])
    
    bootstrap_scores = np.array(bootstrap_scores)
    
    # Compute percentile intervals
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_results = []
    for i, result in enumerate(results):
        lower = np.percentile(bootstrap_scores[:, i], lower_percentile)
        upper = np.percentile(bootstrap_scores[:, i], upper_percentile)
        
        ci_results.append({
            **result,
            f'{metric_key}_ci_lower': float(lower),
            f'{metric_key}_ci_upper': float(upper),
            f'{metric_key}_ci_width': float(upper - lower)
        })
    
    return ci_results