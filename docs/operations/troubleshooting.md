# Troubleshooting

Use `/health` for process liveness and the administrator status page for
dependency reachability. A catalogue error can mean MLflow is down, model
discovery is unconfigured, or governance approval could not be checked.
Confirm the service identity and model version mapping.

For load errors, check the approved package digest, expected export,
preprocessing configuration, framework extra, and configured device. For S3,
confirm the prefix and instance permissions. Failed uploads return a request ID
when the prediction handler runs. Use this ID to correlate stored history and
container logs.
