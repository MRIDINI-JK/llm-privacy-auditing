#!/usr/bin/env python3
"""
Check audit results and enforce privacy policies in CI/CD.
"""
import argparse
import json
import sys

def check_audit_results(audit_file, fail_on_high_risk=False, max_risk_rate=0.3):
    """Check audit results and exit with error code if policies violated"""
    
    with open(audit_file) as f:
        report = json.load(f)
    
    compliance = report['compliance_summary']
    
    print(f"\n{'='*60}")
    print("  PRIVACY AUDIT POLICY CHECK")
    print(f"{'='*60}\n")
    
    # Check compliance score
    compliance_score = compliance['overall_score']
    print(f"Compliance Score: {compliance_score:.1f}%")
    
    if compliance_score < 80:
        print("❌ FAIL: Compliance score below 80%")
        if fail_on_high_risk:
            sys.exit(1)
    else:
        print("✓ PASS: Compliance score meets threshold")
    
    # Check requirements
    not_met = compliance['not_met']
    if not_met > 0:
        print(f"❌ FAIL: {not_met} requirements not met")
        if fail_on_high_risk:
            sys.exit(1)
    else:
        print("✓ PASS: All requirements met or partially met")
    
    print(f"\n{'='*60}\n")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit_file", required=True)
    parser.add_argument("--fail_on_high_risk", action="store_true")
    parser.add_argument("--max_risk_rate", type=float, default=0.3)
    
    args = parser.parse_args()
    check_audit_results(args.audit_file, args.fail_on_high_risk, args.max_risk_rate)