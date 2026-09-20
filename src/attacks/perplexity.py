import numpy as np
from tqdm import tqdm
import torch

def compute_perplexity(model, tokenizer, text, device="cuda"):
    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings.input_ids)
        loss = outputs.loss
    
    return torch.exp(loss).item()

def perplexity_attack(model, tokenizer, samples, device="cuda"):
    results = []
    
    for sample in tqdm(samples, desc="Computing perplexities"):
        ppl = compute_perplexity(model, tokenizer, sample['text'], device)
        results.append({**sample, 'perplexity': ppl})
    
    # Compute risk scores
    perplexities = np.array([r['perplexity'] for r in results])
    ppl_min, ppl_max = perplexities.min(), perplexities.max()
    
    for i, result in enumerate(results):
        normalized = (perplexities[i] - ppl_min) / (ppl_max - ppl_min + 1e-8)
        result['risk_score'] = 1.0 - normalized
    
    return results