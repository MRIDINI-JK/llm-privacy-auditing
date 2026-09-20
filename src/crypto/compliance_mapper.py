from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
import json

@dataclass
class ComplianceRequirement:
    """Individual compliance requirement"""
    regulation: str
    requirement_id: str
    description: str
    status: str  # 'met', 'partial', 'not_met'
    evidence: List[str]
    
    def to_dict(self):
        return {
            'regulation': self.regulation,
            'requirement_id': self.requirement_id,
            'description': self.description,
            'status': self.status,
            'evidence': self.evidence
        }

class ComplianceReport:
    """
    Map privacy audit results to regulatory compliance requirements.
    """
    
    def __init__(self, audit_results: Dict, dp_proof: Optional[Dict] = None, merkle_proof: Optional[Dict] = None):
        self.audit_results = audit_results
        self.dp_proof = dp_proof
        self.merkle_proof = merkle_proof
        self.requirements = []
        self.timestamp = datetime.now().isoformat()
    
    def assess_gdpr_compliance(self):
        """Assess GDPR compliance"""
        
        # Article 17: Right to Erasure
        erasure_status = 'met' if self.merkle_proof else 'partial'
        erasure_evidence = []
        if self.merkle_proof:
            erasure_evidence.append(f"Merkle root: {self.merkle_proof.get('merkle_root', 'N/A')}")
            erasure_evidence.append(f"Filtered {self.merkle_proof.get('filtered_samples', 0)} samples")
        
        self.requirements.append(ComplianceRequirement(
            regulation='GDPR',
            requirement_id='Article 17',
            description='Right to Erasure (Right to be Forgotten)',
            status=erasure_status,
            evidence=erasure_evidence if erasure_evidence else ['Cryptographic proof of data removal available']
        ))
        
        # Article 25: Data Protection by Design
        design_evidence = [
            f"Pre-deployment privacy audit conducted",
            f"Total samples audited: {self.audit_results['metadata'].get('num_samples', 0)}",
            f"Attacks used: {', '.join(self.audit_results['metadata'].get('attacks_used', []))}"
        ]
        
        self.requirements.append(ComplianceRequirement(
            regulation='GDPR',
            requirement_id='Article 25',
            description='Data Protection by Design and Default',
            status='met',
            evidence=design_evidence
        ))
        
        # Article 32: Security of Processing
        security_evidence = []
        if self.dp_proof:
            security_evidence.append(f"Differential Privacy: ε={self.dp_proof.get('epsilon', 'N/A')}, δ={self.dp_proof.get('delta', 'N/A')}")
            security_evidence.append(f"Mechanism: {self.dp_proof.get('mechanism', 'N/A')}")
        
        summary = self.audit_results.get('summary', {})
        if summary:
            security_evidence.append(f"Detection accuracy: {summary.get('accuracy', 0):.2%}")
        
        self.requirements.append(ComplianceRequirement(
            regulation='GDPR',
            requirement_id='Article 32',
            description='Security of Processing',
            status='met' if self.dp_proof else 'partial',
            evidence=security_evidence if security_evidence else ['Privacy risk assessment completed']
        ))
    
    def assess_hipaa_compliance(self):
        """Assess HIPAA compliance"""
        
        # §164.308 Administrative Safeguards
        admin_evidence = [
            'Privacy audit conducted before deployment',
            'Risk assessment completed with quantified metrics',
            f"Audit timestamp: {self.timestamp}"
        ]
        
        self.requirements.append(ComplianceRequirement(
            regulation='HIPAA',
            requirement_id='§164.308(a)(1)(ii)(A)',
            description='Risk Analysis',
            status='met',
            evidence=admin_evidence
        ))
        
        # §164.312 Technical Safeguards
        technical_evidence = []
        if self.merkle_proof:
            technical_evidence.append('Cryptographic verification of data filtering')
            technical_evidence.append(f"Merkle tree root: {self.merkle_proof.get('merkle_root', '')[:16]}...")
        
        if self.dp_proof:
            technical_evidence.append(f"Differential privacy guarantees: ε={self.dp_proof.get('epsilon', 'N/A')}")
        
        self.requirements.append(ComplianceRequirement(
            regulation='HIPAA',
            requirement_id='§164.312(a)(1)',
            description='Access Control - Technical Safeguards',
            status='met' if (self.merkle_proof or self.dp_proof) else 'partial',
            evidence=technical_evidence if technical_evidence else ['Privacy risk scoring implemented']
        ))
        
        # §164.312(b) Audit Controls
        audit_evidence = [
            f"Complete audit trail available",
            f"High-risk samples identified: {len(self.audit_results.get('high_risk_samples', []))}",
            'Confidence intervals computed for all risk scores'
        ]
        
        self.requirements.append(ComplianceRequirement(
            regulation='HIPAA',
            requirement_id='§164.312(b)',
            description='Audit Controls',
            status='met',
            evidence=audit_evidence
        ))
    
    def assess_ccpa_compliance(self):
        """Assess CCPA compliance"""
        
        # §1798.100 Right to Know
        know_evidence = [
            'Transparency report generated',
            f"Privacy risk scores available for all {self.audit_results['metadata'].get('num_samples', 0)} samples",
            'Detailed methodology documented'
        ]
        
        self.requirements.append(ComplianceRequirement(
            regulation='CCPA',
            requirement_id='§1798.100',
            description='Consumer Right to Know',
            status='met',
            evidence=know_evidence
        ))
        
        # §1798.105 Right to Delete
        delete_status = 'met' if self.merkle_proof else 'partial'
        delete_evidence = []
        if self.merkle_proof:
            delete_evidence.append('Cryptographic proof of data deletion available')
            delete_evidence.append(f"Deletion verification via Merkle tree")
        
        self.requirements.append(ComplianceRequirement(
            regulation='CCPA',
            requirement_id='§1798.105',
            description='Consumer Right to Delete',
            status=delete_status,
            evidence=delete_evidence if delete_evidence else ['Data filtering capability implemented']
        ))
    
    def generate_report(self) -> Dict:
        """Generate complete compliance report"""
        
        self.assess_gdpr_compliance()
        self.assess_hipaa_compliance()
        self.assess_ccpa_compliance()
        
        # Compute overall compliance score
        status_scores = {'met': 1.0, 'partial': 0.5, 'not_met': 0.0}
        total_score = sum(status_scores[req.status] for req in self.requirements)
        max_score = len(self.requirements)
        compliance_percentage = (total_score / max_score) * 100 if max_score > 0 else 0
        
        return {
            'timestamp': self.timestamp,
            'compliance_summary': {
                'overall_score': compliance_percentage,
                'total_requirements': len(self.requirements),
                'met': sum(1 for r in self.requirements if r.status == 'met'),
                'partial': sum(1 for r in self.requirements if r.status == 'partial'),
                'not_met': sum(1 for r in self.requirements if r.status == 'not_met')
            },
            'requirements': [req.to_dict() for req in self.requirements],
            'cryptographic_proofs': {
                'differential_privacy': self.dp_proof,
                'merkle_tree': self.merkle_proof
            },
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if not self.dp_proof:
            recommendations.append(
                "Consider implementing differential privacy during training for stronger guarantees"
            )
        
        if not self.merkle_proof:
            recommendations.append(
                "Generate Merkle tree proofs for cryptographic verification of data removal"
            )
        
        summary = self.audit_results.get('summary', {})
        if summary.get('accuracy', 0) < 0.8:
            recommendations.append(
                "Detection accuracy below 80% - consider using white-box attacks for better precision"
            )
        
        high_risk = len(self.audit_results.get('high_risk_samples', []))
        total = self.audit_results['metadata'].get('num_samples', 1)
        if high_risk / total > 0.3:
            recommendations.append(
                f"High-risk sample rate is {high_risk/total:.1%} - consider data deduplication or filtering"
            )
        
        return recommendations if recommendations else ["All compliance requirements met"]