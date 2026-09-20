from examples.api_client_example import PrivacyAuditClient

client = PrivacyAuditClient()
job_id = client.submit_audit("gpt2", max_samples=1)
results = client.get_results(job_id)