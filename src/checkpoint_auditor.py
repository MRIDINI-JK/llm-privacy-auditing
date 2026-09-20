"""
Model Checkpoint Auditor - Track memorization during training.
"""
import os
import json
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

class CheckpointAuditor:
    """
    Audit model checkpoints during training to track memorization drift.
    """
    
    def __init__(self, output_dir: str = "outputs/checkpoint_audits"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audit_history = []
    
    def audit_checkpoint(
        self,
        checkpoint_path: str,
        eval_samples: List[Dict],
        epoch: int,
        step: int,
        tokenizer_name: str = None
    ) -> Dict:
        """
        Audit a single checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            eval_samples: Evaluation samples for MIA
            epoch: Training epoch
            step: Training step
            tokenizer_name: Tokenizer to use
        
        Returns:
            Audit results dict
        """
        from src.utils.model_loader import load_model_and_tokenizer
        from src.attacks.perplexity import perplexity_attack
        from src.attacks.ensemble import ensemble_attack
        
        print(f"\n[Checkpoint Audit] Epoch {epoch}, Step {step}")
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        if tokenizer_name:
            model, tokenizer = load_model_and_tokenizer(checkpoint_path, device="cpu")
        else:
            model = torch.load(checkpoint_path, map_location="cpu")
            tokenizer = None
        
        # Run basic perplexity attack
        results = perplexity_attack(model, tokenizer, eval_samples, device="cpu")
        
        # Compute statistics
        perplexities = [r['perplexity'] for r in results]
        risk_scores = [r['risk_score'] for r in results]
        
        audit_result = {
            'checkpoint': checkpoint_path,
            'epoch': epoch,
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'stats': {
                'mean_perplexity': float(sum(perplexities) / len(perplexities)),
                'max_perplexity': float(max(perplexities)),
                'min_perplexity': float(min(perplexities)),
                'mean_risk_score': float(sum(risk_scores) / len(risk_scores)),
                'high_risk_count': sum(1 for r in risk_scores if r > 0.7),
                'high_risk_rate': sum(1 for r in risk_scores if r > 0.7) / len(risk_scores)
            }
        }
        
        self.audit_history.append(audit_result)
        
        # Save checkpoint-specific results
        checkpoint_name = Path(checkpoint_path).stem
        result_path = self.output_dir / f"{checkpoint_name}_epoch{epoch}_step{step}.json"
        with open(result_path, 'w') as f:
            json.dump(audit_result, f, indent=2)
        
        print(f"✓ Audit complete: {audit_result['stats']}")
        
        return audit_result
    
    def save_history(self):
        """Save complete audit history"""
        history_path = self.output_dir / "audit_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.audit_history, f, indent=2)
        
        # Also save as CSV for easy plotting
        df = pd.DataFrame([
            {
                'epoch': h['epoch'],
                'step': h['step'],
                'timestamp': h['timestamp'],
                **h['stats']
            }
            for h in self.audit_history
        ])
        df.to_csv(self.output_dir / "audit_history.csv", index=False)
        
        print(f"✓ Audit history saved: {history_path}")
    
    def generate_report(self) -> Dict:
        """Generate summary report across all checkpoints"""
        if not self.audit_history:
            return {'error': 'No audit history available'}
        
        report = {
            'total_checkpoints': len(self.audit_history),
            'checkpoints': self.audit_history,
            'trends': self._compute_trends(),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _compute_trends(self) -> Dict:
        """Compute trends in memorization over training"""
        if len(self.audit_history) < 2:
            return {}
        
        risk_rates = [h['stats']['high_risk_rate'] for h in self.audit_history]
        perplexities = [h['stats']['mean_perplexity'] for h in self.audit_history]
        
        return {
            'risk_rate_trend': 'increasing' if risk_rates[-1] > risk_rates[0] else 'decreasing',
            'perplexity_trend': 'increasing' if perplexities[-1] > perplexities[0] else 'decreasing',
            'initial_risk_rate': risk_rates[0],
            'final_risk_rate': risk_rates[-1],
            'max_risk_rate': max(risk_rates),
            'min_risk_rate': min(risk_rates)
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on audit history"""
        recommendations = []
        trends = self._compute_trends()
        
        if trends.get('final_risk_rate', 0) > 0.5:
            recommendations.append(
                "High memorization risk detected - consider applying data deduplication"
            )
        
        if trends.get('risk_rate_trend') == 'increasing':
            recommendations.append(
                "Memorization increasing over training - consider early stopping or DP-SGD"
            )
        
        return recommendations


# Training integration example
class PrivacyAwareTrainer:
    """
    Wrapper around HuggingFace Trainer with privacy auditing.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_samples: List[Dict],
        audit_interval: int = 500
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_samples = eval_samples
        self.audit_interval = audit_interval
        self.auditor = CheckpointAuditor()
        self.step = 0
    
    def on_save_callback(self, checkpoint_path: str, epoch: int):
        """Callback triggered when checkpoint is saved"""
        if self.step % self.audit_interval == 0:
            self.auditor.audit_checkpoint(
                checkpoint_path=checkpoint_path,
                eval_samples=self.eval_samples,
                epoch=epoch,
                step=self.step,
                tokenizer_name=None
            )
    
    def train(self, num_epochs: int):
        """Training loop with privacy auditing"""
        from transformers import Trainer, TrainingArguments
        
        training_args = TrainingArguments(
            output_dir="./checkpoints",
            num_train_epochs=num_epochs,
            save_strategy="steps",
            save_steps=self.audit_interval,
            evaluation_strategy="steps",
            eval_steps=self.audit_interval,
            logging_steps=100
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            tokenizer=self.tokenizer
        )
        
        # Train with auditing
        trainer.train()
        
        # Save final audit history
        self.auditor.save_history()
        report = self.auditor.generate_report()
        
        with open("outputs/checkpoint_audits/final_report.json", 'w') as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--eval_data", required=True)
    parser.add_argument("--output_dir", default="outputs/checkpoint_audits")
    
    args = parser.parse_args()
    
    from src.utils.data_loader import load_custom_dataset
    
    # Load evaluation samples
    eval_samples = load_custom_dataset(args.eval_data)
    
    # Audit all checkpoints in directory
    auditor = CheckpointAuditor(args.output_dir)
    
    checkpoint_files = sorted(Path(args.checkpoint_dir).glob("checkpoint-*"))
    
    for ckpt in checkpoint_files:
        # Extract epoch/step from filename
        parts = ckpt.stem.split('-')
        step = int(parts[-1])
        epoch = step // 1000  # Approximate
        
        auditor.audit_checkpoint(
            checkpoint_path=str(ckpt),
            eval_samples=eval_samples[:50],  # Sample for speed
            epoch=epoch,
            step=step
        )
    
    auditor.save_history()
    report = auditor.generate_report()
    
    print("\n=== CHECKPOINT AUDIT REPORT ===")
    print(json.dumps(report, indent=2))