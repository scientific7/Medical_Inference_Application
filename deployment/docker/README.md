# Containers

The root Dockerfile provides separate backend and frontend targets. The root
Compose file runs both services with a persistent backend data volume and a
read-only approved package mount. See the [Docker guide](../../docs/deployment/docker-deployment.md)
for configuration and validation commands.
