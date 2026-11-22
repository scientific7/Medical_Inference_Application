# Deployment scripts

The EC2 bootstrap installs `deploy.sh` on the host. It pulls digest-pinned
images, retrieves backend settings from Secrets Manager, replaces the
containers, and checks their health. Run `bash -n deploy.sh` for syntax
validation. See the [rollback procedure](../../docs/deployment/aws-deployment.md#rolling-back).
