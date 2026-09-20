import torch
import numpy as np
from tqdm import tqdm

def compute_token_loss(model, tokenizer, text, device="cpu"):
    """Compute per-token loss (more granular than perplexity)"""
    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings.input_ids)
        # Get per-token losses
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = encodings.input_ids[..., 1:].contiguous()
        
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
    
    return token_losses.cpu().numpy()

def token_loss_attack(model, tokenizer, samples, device="cpu"):
    """Attack using mean token loss"""
    results = []
    
    for sample in tqdm(samples, desc="Token loss attack"):
        losses = compute_token_loss(model, tokenizer, sample['text'], device)
        mean_loss = float(np.mean(losses))
        
        results.append({
            **sample,
            'token_loss': mean_loss,
            'token_losses': losses.tolist()  # Store for min-k%
        })
    
    # Normalize to risk scores
    losses = np.array([r['token_loss'] for r in results])
    loss_min, loss_max = losses.min(), losses.max()
    
    for i, result in enumerate(results):
        normalized = (losses[i] - loss_min) / (loss_max - loss_min + 1e-8)
        result['token_loss_risk'] = 1.0 - normalized
    
    return results