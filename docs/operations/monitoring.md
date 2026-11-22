# Monitoring

The EC2 deployment sends container output to the CloudWatch log group.
Inference events carry the request ID, model identity, task, outcome,
device, and latency. Logs exclude input bytes and prediction contents.
Use the admin status endpoint for dependency reachability and the system
metrics endpoint for recorded request counts and latency. The EC2 alarm
reports failed host status checks. The ALB monitors frontend health.

Operational metrics describe application availability. Estimating model
performance on the deployment population requires appropriate ground-truth
labels.
