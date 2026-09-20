import torch
import numpy as np
from tqdm import tqdm

def compute_perturbed_loss(model, tokenizer, text, num_perturbations=5, device="cpu"):
    """
    Neighborhood attack: Compare loss on original vs. perturbed versions.
    Members have lower variance in loss.
    """
    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)
    
    # Original loss
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings.input_ids)
        original_loss = outputs.loss.item()
    
    # Perturbed losses (randomly drop tokens)
    perturbed_losses = []
    tokens = encodings.input_ids[0].tolist()
    
    for _ in range(num_perturbations):
        if len(tokens) <= 5:
            break
            
        # Drop 10% of tokens randomly
        keep_indices = np.random.choice(
            len(tokens),
            size=int(len(tokens) * 0.9),
            replace=False
        )
        perturbed_tokens = [tokens[i] for i in sorted(keep_indices)]
        
        perturbed_encodings = {
            'input_ids': torch.tensor([perturbed_tokens]).to(device),
            'attention_mask': torch.ones(1, len(perturbed_tokens)).to(device)
        }
        
        with torch.no_grad():
            outputs = model(**perturbed_encodings, labels=perturbed_encodings['input_ids'])
            perturbed_losses.append(outputs.loss.item())
    
    loss_variance = float(np.var(perturbed_losses)) if perturbed_losses else 0.0
    
    return original_loss, loss_variance

def neighborhood_attack(model, tokenizer, samples, device="cpu"):
    """Attack using loss variance under perturbations"""
    results = []
    
    for sample in tqdm(samples, desc="Neighborhood attack"):
        orig_loss, variance = compute_perturbed_loss(
            model, tokenizer, sample['text'], device=device
        )
        
        results.append({
            **sample,
            'neighborhood_loss': orig_loss,
            'neighborhood_variance': variance
        })
    
    # Normalize: Lower variance = member
    variances = np.array([r['neighborhood_variance'] for r in results])
    var_min, var_max = variances.min(), variances.max()
    
    for i, result in enumerate(results):
        normalized = (variances[i] - var_min) / (var_max - var_min + 1e-8)
        result['neighborhood_risk'] = 1.0 - normalized
    
    return results