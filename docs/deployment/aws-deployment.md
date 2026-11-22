# AWS deployment

The Terraform root in `deployment/terraform` creates a private EC2 host,
instance profile, restricted security group, CloudWatch log group, EC2
status alarm, and registration with an existing ALB target group. It uses
existing VPC, private subnet, HTTPS ingress, S3 bucket, and PostgreSQL
infrastructure. The backend is reachable on the Docker network. The ALB
forwards Streamlit traffic to port 8501 and checks `/_stcore/health`.
Configure the ALB ingress access policy and TLS certificate outside this
root. Allow WebSocket connections for Streamlit.

Use an Amazon Linux 2023 AMI with matching image architecture and AWS CLI.
Provide private subnet connectivity to the package and approved container
registries, AWS services, database, governance, MLflow, and identity provider.
Supply the corresponding TCP destinations as `egress_rules`. Use image
references pinned by digest and pullable by the Docker host. Private image
registries require host-side authentication provisioned separately.

Create the inference database and run the documented migration with the
migration role. Store the application's Docker environment file contents as
a Secrets Manager string. Include the PostgreSQL URL with TLS verification,
OIDC configuration, and scoped governance credentials. The instance role
reads that secret at startup. Configure customer-managed KMS key access
separately when the secret or approved artifacts use such a key. Backend
environment values can be inspected by privileged host operators. Limit
Systems Manager access accordingly.

Review Terraform changes with `terraform init`, `terraform validate`, and
`terraform plan` from the Terraform directory. Apply only in an authorized
environment. Use a restricted remote state backend for shared deployments.
Container logs go to the deployment CloudWatch log group. Optional alarm
destinations can be supplied with `alarm_actions`.

## Rolling back

Record the previous backend and frontend digests before switching versions.
Use Systems Manager to run `/usr/local/sbin/deploy-medical-imaging` with the
same deployment arguments and the previous image digests. Validate the ALB
target health and the research workflow. Update the desired Terraform image
references afterward. Replacing containers interrupts active inference and
discards process-local batches. Schedule the switch for a maintenance window
and check database schema compatibility before rollback.
