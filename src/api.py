"""
REST API for LLM Privacy Auditing Service.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
import json
import uuid
from pathlib import Path
import pandas as pd

app = FastAPI(
    title="LLM Privacy Audit API",
    description="REST API for pre-deployment privacy auditing of LLMs",
    version="1.0.0"
)

# Data models
class AuditRequest(BaseModel):
    model_name: str
    max_samples: int = 50
    risk_threshold: float = 0.7
    attacks: List[str] = ["perplexity", "token_loss", "mink_prob", "neighborhood"]
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5

class AuditStatus(BaseModel):
    job_id: str
    status: str  # pending, running, completed, failed
    progress: float
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None

class AuditResult(BaseModel):
    job_id: str
    model_name: str
    num_samples: int
    high_risk_count: int
    compliance_score: float
    filter_rate: float
    mean_risk_score: float
    timestamp: str

# In-memory job storage (use Redis/DB in production)
jobs = {}

# Background task for running audits
def run_audit_task(job_id: str, request: AuditRequest):
    """Background task to run privacy audit"""
    from src.utils.model_loader import load_model_and_tokenizer
    from src.utils.data_loader import load_tofu_dataset
    from src.attacks.perplexity import perplexity_attack
    from src.attacks.token_loss import token_loss_attack
    from src.attacks.mink_prob import mink_prob_attack
    from src.attacks.neighborhood import neighborhood_attack
    from src.attacks.ensemble import ensemble_attack
    from src.crypto.dp_proof import generate_dp_proof
    from src.crypto.compliance_mapper import ComplianceReport
    
    try:
        jobs[job_id]['status'] = 'running'
        jobs[job_id]['progress'] = 0.1
        
        # Load model
        model, tokenizer = load_model_and_tokenizer(request.model_name, device="cpu")
        jobs[job_id]['progress'] = 0.2
        
        # Load data
        samples = load_tofu_dataset(max_samples=request.max_samples)
        jobs[job_id]['progress'] = 0.3
        
        # Run attacks
        all_results = []
        
        if 'perplexity' in request.attacks:
            results = perplexity_attack(model, tokenizer, samples, device="cpu")
            all_results.append(results)
            jobs[job_id]['progress'] = 0.4
        
        if 'token_loss' in request.attacks:
            results = token_loss_attack(model, tokenizer, samples, device="cpu")
            all_results.append(results)
            jobs[job_id]['progress'] = 0.5
        
        if 'mink_prob' in request.attacks:
            results = mink_prob_attack(model, tokenizer, samples, device="cpu")
            all_results.append(results)
            jobs[job_id]['progress'] = 0.6
        
        if 'neighborhood' in request.attacks:
            results = neighborhood_attack(model, tokenizer, samples, device="cpu")
            all_results.append(results)
            jobs[job_id]['progress'] = 0.7
        
        # Ensemble
        final_results = ensemble_attack(all_results)
        jobs[job_id]['progress'] = 0.8
        
        # Generate proofs
        high_risk = [r for r in final_results if r['ensemble_risk'] >= request.risk_threshold]
        dp_proof = generate_dp_proof(
            high_risk, final_results,
            epsilon=request.dp_epsilon,
            delta=request.dp_delta
        )
        jobs[job_id]['progress'] = 0.9
        
        # Compliance
        audit_report = {
            'metadata': {
                'model': request.model_name,
                'timestamp': datetime.now().isoformat(),
                'num_samples': len(samples),
                'attacks_used': request.attacks
            },
            'high_risk_samples': high_risk,
            'privacy_loss': {
                'filter_rate': len(high_risk) / len(samples),
                'mean_risk_score': sum(r['ensemble_risk'] for r in final_results) / len(final_results)
            }
        }
        
        compliance = ComplianceReport(
            audit_report,
            dp_proof.to_dict(),
            None
        )
        compliance_report = compliance.generate_report()
        
        # Save results
        output_dir = Path(f"outputs/api_jobs/{job_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "audit_report.json", 'w') as f:
            json.dump(audit_report, f, indent=2)
        
        with open(output_dir / "compliance_report.json", 'w') as f:
            json.dump(compliance_report, f, indent=2)
        
        pd.DataFrame(final_results).to_csv(output_dir / "results.csv", index=False)
        
        # Update job status
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['progress'] = 1.0
        jobs[job_id]['completed_at'] = datetime.now().isoformat()
        jobs[job_id]['results'] = {
            'num_samples': len(samples),
            'high_risk_count': len(high_risk),
            'compliance_score': compliance_report['compliance_summary']['overall_score'],
            'filter_rate': len(high_risk) / len(samples),
            'mean_risk_score': audit_report['privacy_loss']['mean_risk_score']
        }
        
    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)
        jobs[job_id]['completed_at'] = datetime.now().isoformat()


# API Endpoints

@app.get("/")
def root():
    """API health check"""
    return {
        "service": "LLM Privacy Audit API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.post("/audit", response_model=AuditStatus)
def create_audit(request: AuditRequest, background_tasks: BackgroundTasks):
    """
    Submit a new privacy audit job.
    """
    job_id = str(uuid.uuid4())
    
    jobs[job_id] = {
        'job_id': job_id,
        'status': 'pending',
        'progress': 0.0,
        'created_at': datetime.now().isoformat(),
        'request': request.dict()
    }
    
    # Run audit in background
    background_tasks.add_task(run_audit_task, job_id, request)
    
    return AuditStatus(
        job_id=job_id,
        status='pending',
        progress=0.0,
        created_at=jobs[job_id]['created_at']
    )

@app.get("/audit/{job_id}", response_model=AuditStatus)
def get_audit_status(job_id: str):
    """
    Get status of an audit job.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    return AuditStatus(
        job_id=job_id,
        status=job['status'],
        progress=job['progress'],
        created_at=job['created_at'],
        completed_at=job.get('completed_at'),
        error=job.get('error')
    )

@app.get("/audit/{job_id}/results", response_model=AuditResult)
def get_audit_results(job_id: str):
    """
    Get results of completed audit.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job['status'] != 'completed':
        raise HTTPException(status_code=400, detail=f"Job status: {job['status']}")
    
    results = job['results']
    
    return AuditResult(
        job_id=job_id,
        model_name=job['request']['model_name'],
        num_samples=results['num_samples'],
        high_risk_count=results['high_risk_count'],
        compliance_score=results['compliance_score'],
        filter_rate=results['filter_rate'],
        mean_risk_score=results['mean_risk_score'],
        timestamp=job['completed_at']
    )

@app.get("/audit/{job_id}/download/{file_type}")
def download_audit_file(job_id: str, file_type: str):
    """
    Download audit files (report, compliance, results).
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if jobs[job_id]['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Job not completed")
    
    output_dir = Path(f"outputs/api_jobs/{job_id}")
    
    file_map = {
        'report': output_dir / "audit_report.json",
        'compliance': output_dir / "compliance_report.json",
        'results': output_dir / "results.csv"
    }
    
    if file_type not in file_map:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    file_path = file_map[file_type]
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type='application/octet-stream'
    )

@app.delete("/audit/{job_id}")
def delete_audit(job_id: str):
    """
    Delete an audit job and its results.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete files
    output_dir = Path(f"outputs/api_jobs/{job_id}")
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    
    # Delete job record
    del jobs[job_id]
    
    return {"message": "Job deleted successfully"}

@app.get("/jobs")
def list_jobs(status: Optional[str] = None):
    """
    List all audit jobs, optionally filtered by status.
    """
    filtered_jobs = []
    
    for job_id, job in jobs.items():
        if status is None or job['status'] == status:
            filtered_jobs.append({
                'job_id': job_id,
                'status': job['status'],
                'created_at': job['created_at'],
                'model': job['request']['model_name']
            })
    
    return {'jobs': filtered_jobs}


if __name__ == "__main__":
    import uvicorn
    
    print(f"\n{'='*60}")
    print("  LLM PRIVACY AUDIT API")
    print(f"{'='*60}\n")
    print("Starting API server at http://localhost:8000")
    print("API docs available at http://localhost:8000/docs\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)