import torch
import numpy as np
from tqdm import tqdm

def compute_mink_prob(model, tokenizer, text, k_percent=0.2, device="cpu"):
    """
    Min-k% probability: average of lowest k% token probabilities.
    Better than perplexity for membership inference.
    
    Paper: https://arxiv.org/abs/2310.16789
    """
    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings.input_ids)
        logits = outputs.logits
        
        # Get probabilities for actual tokens
        probs = torch.softmax(logits, dim=-1)
        shift_probs = probs[..., :-1, :].contiguous()
        shift_labels = encodings.input_ids[..., 1:].contiguous()
        
        # Extract probability of actual next token
        token_probs = torch.gather(
            shift_probs.view(-1, shift_probs.size(-1)),
            1,
            shift_labels.view(-1, 1)
        ).squeeze()
    
    token_probs = token_probs.cpu().numpy()
    
    # Get lowest k% probabilities
    k = max(1, int(len(token_probs) * k_percent))
    mink_probs = np.partition(token_probs, k)[:k]
    
    return float(np.mean(mink_probs))

def mink_prob_attack(model, tokenizer, samples, k_percent=0.2, device="cpu"):
    """Attack using min-k% probability"""
    results = []
    
    for sample in tqdm(samples, desc="Min-k% prob attack"):
        mink = compute_mink_prob(
            model, tokenizer, sample['text'], k_percent, device
        )
        
        results.append({**sample, 'mink_prob': mink})
    
    # Normalize to risk scores (higher prob = member)
    probs = np.array([r['mink_prob'] for r in results])
    prob_min, prob_max = probs.min(), probs.max()
    
    for i, result in enumerate(results):
        normalized = (probs[i] - prob_min) / (prob_max - prob_min + 1e-8)
        result['mink_prob_risk'] = normalized  # Higher is more member-like
    
    return results