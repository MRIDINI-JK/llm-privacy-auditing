from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
import numpy as np

def evaluate_attack(results, threshold=0.5, score_key='ensemble_risk'):
    """Enhanced evaluation with PR-AUC"""
    labeled = [r for r in results if r.get('is_member') is not None]
    
    y_true = np.array([r['is_member'] for r in labeled])
    y_scores = np.array([r[score_key] for r in labeled])
    y_pred = (y_scores >= threshold).astype(int)
    
    # ROC metrics
    roc_auc = roc_auc_score(y_true, y_scores)
    
    # PR metrics
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall_vals, precision_vals)
    
    # Threshold-based
    accuracy = (y_pred == y_true).mean()
    precision = ((y_pred == 1) & (y_true == 1)).sum() / max((y_pred == 1).sum(), 1)
    recall = ((y_pred == 1) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def find_optimal_threshold(results, score_key='ensemble_risk'):
    """Find threshold maximizing Youden's J statistic"""
    labeled = [r for r in results if r.get('is_member') is not None]
    y_true = np.array([r['is_member'] for r in labeled])
    y_scores = np.array([r[score_key] for r in labeled])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    return float(thresholds[np.argmax(j_scores)])