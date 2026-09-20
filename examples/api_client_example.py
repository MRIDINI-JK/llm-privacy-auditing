"""
Example API client for LLM Privacy Audit Service.
"""
import requests
import time
from typing import Dict, Optional

class PrivacyAuditClient:
    """
    Client for interacting with Privacy Audit API.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def submit_audit(
        self,
        model_name: str,
        max_samples: int = 50,
        risk_threshold: float = 0.7,
        attacks: list = None,
        dp_epsilon: float = 1.0,
        dp_delta: float = 1e-5
    ) -> str:
        """
        Submit a new audit job.
        
        Returns:
            job_id
        """
        if attacks is None:
            attacks = ["perplexity", "token_loss", "mink_prob", "neighborhood"]
        
        response = requests.post(
            f"{self.base_url}/audit",
            json={
                "model_name": model_name,
                "max_samples": max_samples,
                "risk_threshold": risk_threshold,
                "attacks": attacks,
                "dp_epsilon": dp_epsilon,
                "dp_delta": dp_delta
            }
        )
        
        response.raise_for_status()
        return response.json()['job_id']
    
    def get_status(self, job_id: str) -> Dict:
        """Get job status"""
        response = requests.get(f"{self.base_url}/audit/{job_id}")
        response.raise_for_status()
        return response.json()
    
    def wait_for_completion(self, job_id: str, poll_interval: int = 5) -> Dict:
        """
        Wait for job to complete and return results.
        
        Args:
            job_id: Job ID
            poll_interval: Seconds between status checks
        
        Returns:
            Final job status
        """
        print(f"Waiting for job {job_id} to complete...")
        
        while True:
            status = self.get_status(job_id)
            
            print(f"Status: {status['status']} ({status['progress']*100:.0f}%)")
            
            if status['status'] == 'completed':
                print("✓ Audit completed!")
                return status
            
            if status['status'] == 'failed':
                print(f"✗ Audit failed: {status.get('error')}")
                return status
            
            time.sleep(poll_interval)
    
    def get_results(self, job_id: str) -> Dict:
        """Get audit results"""
        response = requests.get(f"{self.base_url}/audit/{job_id}/results")
        response.raise_for_status()
        return response.json()
    
    def download_file(self, job_id: str, file_type: str, output_path: str):
        """
        Download audit file.
        
        Args:
            job_id: Job ID
            file_type: 'report', 'compliance', or 'results'
            output_path: Where to save the file
        """
        response = requests.get(
            f"{self.base_url}/audit/{job_id}/download/{file_type}",
            stream=True
        )
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✓ Downloaded {file_type} to {output_path}")
    
    def list_jobs(self, status: Optional[str] = None) -> list:
        """List all jobs"""
        params = {'status': status} if status else {}
        response = requests.get(f"{self.base_url}/jobs", params=params)
        response.raise_for_status()
        return response.json()['jobs']
    
    def delete_job(self, job_id: str):
        """Delete a job"""
        response = requests.delete(f"{self.base_url}/audit/{job_id}")
        response.raise_for_status()
        print(f"✓ Job {job_id} deleted")


# Example usage
if __name__ == "__main__":
    client = PrivacyAuditClient()
    
    print("="*60)
    print("  PRIVACY AUDIT API CLIENT - EXAMPLE")
    print("="*60)
    
    # Submit audit
    print("\n[1] Submitting audit job...")
    job_id = client.submit_audit(
        model_name="gpt2",
        max_samples=20,
        risk_threshold=0.7
    )
    print(f"✓ Job submitted: {job_id}")
    
    # Wait for completion
    print("\n[2] Waiting for completion...")
    status = client.wait_for_completion(job_id)
    
    if status['status'] == 'completed':
        # Get results
        print("\n[3] Fetching results...")
        results = client.get_results(job_id)
        
        print(f"\n{'='*60}")
        print("  AUDIT RESULTS")
        print(f"{'='*60}")
        print(f"Model: {results['model_name']}")
        print(f"Samples: {results['num_samples']}")
        print(f"High-Risk: {results['high_risk_count']} ({results['filter_rate']:.1%})")
        print(f"Compliance: {results['compliance_score']:.1f}%")
        print(f"Mean Risk: {results['mean_risk_score']:.3f}")
        
        # Download files
        print("\n[4] Downloading reports...")
        client.download_file(job_id, 'report', f'audit_report_{job_id}.json')
        client.download_file(job_id, 'compliance', f'compliance_{job_id}.json')
        client.download_file(job_id, 'results', f'results_{job_id}.csv')
    
    # List all jobs
    print("\n[5] Listing all jobs...")
    jobs = client.list_jobs()
    print(f"Total jobs: {len(jobs)}")
    for job in jobs[:5]:
        print(f"  - {job['job_id'][:8]}: {job['status']} ({job['model']})")