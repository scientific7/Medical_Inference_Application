# Deployment architecture

Local Compose serves the API and workstation on loopback. The AWS
configuration places Streamlit behind an existing HTTPS ALB and keeps
FastAPI on the host's Docker network. The host uses an instance role for
Secrets Manager and approved S3 artifacts. PostgreSQL runs externally.

[Deployment flow](../diagrams/deployment-flow.mmd)
