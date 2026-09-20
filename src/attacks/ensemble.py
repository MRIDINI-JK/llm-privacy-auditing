import numpy as np

def ensemble_attack(results_list, weights=None):
    """
    Combine multiple attack results into ensemble score.
    
    Args:
        results_list: List of result dicts from different attacks
        weights: Dict mapping attack names to weights (default: equal)
    
    Returns:
        Combined results with ensemble_risk_score
    """
    if weights is None:
        weights = {
            'perplexity': 0.25,
            'token_loss': 0.25,
            'mink_prob': 0.30,
            'neighborhood': 0.20
        }
    
    # Merge all results by sample_id
    combined = {}
    for results in results_list:
        for r in results:
            sid = r['sample_id']
            if sid not in combined:
                combined[sid] = {
                    'sample_id': sid,
                    'text': r['text'],
                    'is_member': r.get('is_member')
                }
            combined[sid].update(r)
    
    # Compute ensemble score
    for sid, result in combined.items():
        scores = []
        score_weights = []
        
        if 'risk_score' in result:  # perplexity
            scores.append(result['risk_score'])
            score_weights.append(weights.get('perplexity', 0))
        
        if 'token_loss_risk' in result:
            scores.append(result['token_loss_risk'])
            score_weights.append(weights.get('token_loss', 0))
        
        if 'mink_prob_risk' in result:
            scores.append(result['mink_prob_risk'])
            score_weights.append(weights.get('mink_prob', 0))
        
        if 'neighborhood_risk' in result:
            scores.append(result['neighborhood_risk'])
            score_weights.append(weights.get('neighborhood', 0))
        
        # Weighted average
        if scores:
            score_weights = np.array(score_weights)
            score_weights = score_weights / score_weights.sum()
            result['ensemble_risk'] = float(np.average(scores, weights=score_weights))
        else:
            result['ensemble_risk'] = 0.0
    
    return list(combined.values())