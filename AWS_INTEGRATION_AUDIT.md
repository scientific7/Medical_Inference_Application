# AWS Integration Audit — Medical Inference Application

## 1. Existing AWS Infrastructure & Terraform (`deployment/terraform/main.tf` + Platform AWS)
- **Account / Region:** `2003****8406` (masked), `ap-south-1`, profile `medtriage-admin`.
- **Existing Terraform (`deployment/terraform/main.tf`):** Defines least-privilege EC2 instance behind HTTPS ALB, IMDSv2 required, encrypted `gp3` EBS volume, IAM instance profile with scoped `s3:GetObject`, `secretsmanager:GetSecretValue`, and CloudWatch Logs `/medical-imaging/{name}`.
- **Existing Platform AWS Batch Infrastructure (`ap-south-1`):**
  - Batch Job Queue: `medical-imaging-triage-dev-queue`
  - Inference Worker Job Definition: `medical-imaging-platform-inference-worker:1` (previously verified with live AWS Batch inference job `31a9f5e5-2b59-4941-9da2-d06247dd3f02`, `SUCCEEDED`)
  - S3 Results/Artifacts Bucket: `medical-imaging-triage-results-e09807c9`
  - CloudWatch Log Group: `/aws/batch/medical-imaging-triage-dev`
  - **EC2 GPU G/VT Quota:** `0 vCPUs` in account `2003****8406` — GPU profile (`aws-gpu-g5-xlarge`) is explicitly guarded in the Control Center until AWS approves a GPU quota increase.

## 2. AWS Safety & Developer Auth Rules
- AWS CLI SSO (`medtriage-admin`) is strictly separate from application OIDC authentication.
- Never stores AWS credentials in the GUI, never bypasses MFA, never provisions paid infrastructure automatically.
- Supports automatic graceful fallback to local mock/CPU compute when AWS credentials or SSO are unavailable.
