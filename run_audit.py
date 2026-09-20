import argparse
import pandas as pd
import json
import torch
from datetime import datetime
from src.utils.model_loader import load_model_and_tokenizer
from src.utils.data_loader import load_tofu_dataset, load_custom_dataset
from src.attacks.perplexity import perplexity_attack
from src.attacks.token_loss import token_loss_attack
from src.attacks.mink_prob import mink_prob_attack
from src.attacks.neighborhood import neighborhood_attack
from src.attacks.ensemble import ensemble_attack
from src.utils.confidence import bootstrap_confidence_interval
from src.evaluator import evaluate_attack, find_optimal_threshold
from src.crypto.dp_proof import generate_dp_proof, verify_dp_proof, compute_privacy_loss
from src.crypto.merkle_tree import generate_exclusion_proof
from src.crypto.compliance_mapper import ComplianceReport

def main():
    parser = argparse.ArgumentParser(description="LLM Privacy Audit - Phase 3 (Cryptographic & Compliance)")
    parser.add_argument("--model", type=str, default="gpt2", help="HuggingFace model name")
    parser.add_argument("--dataset", type=str, default="tofu", choices=["tofu", "custom"])
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="outputs/phase3")
    parser.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--risk_threshold", type=float, default=0.7, help="Risk threshold for filtering")
    parser.add_argument("--dp_epsilon", type=float, default=1.0, help="DP privacy budget")
    parser.add_argument("--dp_delta", type=float, default=1e-5, help="DP failure probability")
    parser.add_argument("--attacks", type=str, default="all")
    
    args = parser.parse_args()
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model, device=args.device)
    
    # Load dataset
    if args.dataset == "tofu":
        samples = load_tofu_dataset(split="train", max_samples=args.max_samples)
    else:
        samples = load_custom_dataset(args.data_path)
    
    print(f"\n{'='*60}")
    print(f"  LLM PRIVACY AUDIT - PHASE 3")
    print(f"  Cryptographic & Compliance Layer")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Samples: {len(samples)}")
    print(f"Device: {args.device}")
    print(f"Risk Threshold: {args.risk_threshold}")
    print(f"DP Budget: ε={args.dp_epsilon}, δ={args.dp_delta}\n")
    
    # Run attacks
    attack_list = args.attacks.split(',')
    if 'all' in attack_list:
        attack_list = ['perplexity', 'token_loss', 'mink_prob', 'neighborhood']
    
    all_results = []
    
    if 'perplexity' in attack_list:
        print("[1/4] Running perplexity attack...")
        results = perplexity_attack(model, tokenizer, samples, device=args.device)
        all_results.append(results)
    
    if 'token_loss' in attack_list:
        print("[2/4] Running token loss attack...")
        results = token_loss_attack(model, tokenizer, samples, device=args.device)
        all_results.append(results)
    
    if 'mink_prob' in attack_list:
        print("[3/4] Running min-k% probability attack...")
        results = mink_prob_attack(model, tokenizer, samples, device=args.device)
        all_results.append(results)
    
    if 'neighborhood' in attack_list:
        print("[4/4] Running neighborhood attack...")
        results = neighborhood_attack(model, tokenizer, samples, device=args.device)
        all_results.append(results)
    
    # Ensemble
    print("\n=== Computing ensemble scores ===")
    ensemble_results = ensemble_attack(all_results)
    
    # Confidence intervals
    print("Computing confidence intervals...")
    final_results = bootstrap_confidence_interval(
        ensemble_results,
        metric_key='ensemble_risk',
        n_bootstrap=1000
    )
    
    # Identify high-risk samples
    high_risk_samples = [r for r in final_results if r['ensemble_risk'] >= args.risk_threshold]
    print(f"\n✓ Identified {len(high_risk_samples)} high-risk samples (threshold: {args.risk_threshold})")
    
    # ===== PHASE 3: CRYPTOGRAPHIC PROOFS =====
    
    print(f"\n{'='*60}")
    print("  CRYPTOGRAPHIC PROOF GENERATION")
    print(f"{'='*60}\n")
    
    # 1. Differential Privacy Proof
    print("[1/3] Generating Differential Privacy proof...")
    dp_proof = generate_dp_proof(
        filtered_samples=high_risk_samples,
        total_samples=final_results,
        epsilon=args.dp_epsilon,
        delta=args.dp_delta,
        mechanism="gaussian"
    )
    
    # Verify proof
    is_valid = verify_dp_proof(dp_proof)
    print(f"  ✓ DP Proof generated (ε={dp_proof.epsilon}, δ={dp_proof.delta})")
    print(f"  ✓ Proof verification: {'PASSED' if is_valid else 'FAILED'}")
    print(f"  ✓ Proof hash: {dp_proof.proof_hash[:32]}...")
    
    # 2. Merkle Tree Exclusion Proof
    print("\n[2/3] Generating Merkle tree exclusion proof...")
    merkle_proof = generate_exclusion_proof(
        all_samples=final_results,
        filtered_samples=high_risk_samples
    )
    print(f"  ✓ Merkle root: {merkle_proof['merkle_root'][:32]}...")
    print(f"  ✓ Remaining samples: {merkle_proof['remaining_samples']}")
    print(f"  ✓ Filtered samples: {merkle_proof['filtered_samples']}")
    
    # 3. Privacy Loss Statistics
    print("\n[3/3] Computing privacy loss statistics...")
    privacy_loss = compute_privacy_loss(
        final_results,
        threshold=args.risk_threshold,
        score_key='ensemble_risk'
    )
    print(f"  ✓ Filter rate: {privacy_loss['filter_rate']:.2%}")
    print(f"  ✓ Mean risk score: {privacy_loss['mean_risk_score']:.4f}")
    
    # ===== COMPLIANCE MAPPING =====
    
    print(f"\n{'='*60}")
    print("  REGULATORY COMPLIANCE ASSESSMENT")
    print(f"{'='*60}\n")
    
    # Evaluate
    labeled_count = sum(1 for r in final_results if r.get('is_member') is not None)
    
    audit_report = {
        'metadata': {
            'model': args.model,
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(samples),
            'attacks_used': attack_list,
            'device': args.device
        },
        'summary': {},
        'high_risk_samples': sorted(
            final_results,
            key=lambda x: x['ensemble_risk'],
            reverse=True
        )[:20],
        'privacy_loss': privacy_loss
    }
    
    if labeled_count > 0:
        optimal_threshold = find_optimal_threshold(final_results)
        metrics = evaluate_attack(final_results, threshold=optimal_threshold)
        
        print("Detection Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        audit_report['summary'] = {
            'optimal_threshold': optimal_threshold,
            **metrics
        }
    else:
        audit_report['summary']['note'] = "No ground truth labels available"
    
    # Generate compliance report
    print("\nGenerating compliance report...")
    compliance = ComplianceReport(
        audit_results=audit_report,
        dp_proof=dp_proof.to_dict(),
        merkle_proof=merkle_proof
    )
    
    compliance_report = compliance.generate_report()
    
    print(f"\n✓ Compliance Score: {compliance_report['compliance_summary']['overall_score']:.1f}%")
    print(f"  - Requirements met: {compliance_report['compliance_summary']['met']}")
    print(f"  - Partial compliance: {compliance_report['compliance_summary']['partial']}")
    print(f"  - Not met: {compliance_report['compliance_summary']['not_met']}")
    
    # ===== SAVE OUTPUTS =====
    
    print(f"\n{'='*60}")
    print("  SAVING OUTPUTS")
    print(f"{'='*60}\n")
    
    # 1. CSV with results
    csv_path = f"{args.output_dir}/audit_results.csv"
    df = pd.DataFrame(final_results)
    df.to_csv(csv_path, index=False)
    print(f"✓ CSV saved: {csv_path}")
    
    # 2. Complete audit report
    audit_path = f"{args.output_dir}/audit_report.json"
    with open(audit_path, 'w') as f:
        json.dump(audit_report, f, indent=2)
    print(f"✓ Audit report saved: {audit_path}")
    
    # 3. Compliance report
    compliance_path = f"{args.output_dir}/compliance_report.json"
    with open(compliance_path, 'w') as f:
        json.dump(compliance_report, f, indent=2)
    print(f"✓ Compliance report saved: {compliance_path}")
    
    # 4. Cryptographic proofs
    crypto_path = f"{args.output_dir}/cryptographic_proofs.json"
    crypto_bundle = {
        'differential_privacy': dp_proof.to_dict(),
        'merkle_tree': merkle_proof,
        'verification': {
            'dp_proof_valid': is_valid,
            'timestamp': datetime.now().isoformat()
        }
    }
    with open(crypto_path, 'w') as f:
        json.dump(crypto_bundle, f, indent=2)
    print(f"✓ Cryptographic proofs saved: {crypto_path}")
    
    # 5. Executive summary
    # 5. Executive summary
    summary_path = f"{args.output_dir}/executive_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:  # Added encoding='utf-8'
        f.write("="*70 + "\n")
        f.write("  LLM PRIVACY AUDIT - EXECUTIVE SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Audit Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Samples: {len(samples)}\n")
        f.write(f"High-Risk Samples: {len(high_risk_samples)} ({len(high_risk_samples)/len(samples)*100:.1f}%)\n\n")
    
        f.write("-"*70 + "\n")
        f.write("PRIVACY METRICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Filter Rate: {privacy_loss['filter_rate']:.2%}\n")
        f.write(f"Mean Risk Score: {privacy_loss['mean_risk_score']:.4f}\n")
        f.write(f"Max Risk Score: {privacy_loss['max_risk_score']:.4f}\n\n")
    
        f.write("-"*70 + "\n")
        f.write("CRYPTOGRAPHIC VERIFICATION\n")
        f.write("-"*70 + "\n")
        f.write(f"DP Budget: epsilon={dp_proof.epsilon}, delta={dp_proof.delta}\n")  # Changed ε to 'epsilon'
        f.write(f"DP Mechanism: {dp_proof.mechanism}\n")
        f.write(f"Merkle Root: {merkle_proof['merkle_root'][:32]}...\n")
        f.write(f"Proof Verification: {'PASSED' if is_valid else 'FAILED'}\n\n")
    
        f.write("-"*70 + "\n")
        f.write("REGULATORY COMPLIANCE\n")
        f.write("-"*70 + "\n")
        f.write(f"Overall Score: {compliance_report['compliance_summary']['overall_score']:.1f}%\n")
        f.write(f"Requirements Met: {compliance_report['compliance_summary']['met']}/{compliance_report['compliance_summary']['total_requirements']}\n\n")
    
        f.write("Recommendations:\n")
        for i, rec in enumerate(compliance_report['recommendations'], 1):
            f.write(f"  {i}. {rec}\n")
    
    print(f"✓ Executive summary saved: {summary_path}")
    
    print(f"\n{'='*60}")
    print("  AUDIT COMPLETE")
    print(f"{'='*60}\n")
    print(f"All outputs saved to: {args.output_dir}/")
    print(f"\nNext steps:")
    print(f"  1. Review compliance report for regulatory gaps")
    print(f"  2. Filter high-risk samples from training data")
    print(f"  3. Retrain model or apply machine unlearning")
    print(f"  4. Verify cryptographic proofs with stakeholders\n")

if __name__ == "__main__":
    main()