# Inference host infrastructure

This Terraform root attaches an inference host to existing AWS networking,
storage, and HTTPS ingress. See the [AWS deployment guide](../../docs/deployment/aws-deployment.md)
for inputs, validation, and rollback. Validate with `terraform init -backend=false`
and `terraform validate` before reviewing an environment-specific plan.
