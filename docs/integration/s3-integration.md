# S3 artifact integration

Install the `aws` Python extra and set `INFERENCE_S3_ARTIFACT_ROOT` to
`s3://BUCKET/PREFIX`. The deployment uses the AWS credential provider chain.
Grant the instance role `s3:GetObject` for `PREFIX/*` and constrain
`s3:ListBucket` to the prefix for the administrative health check.

Governance currently approves a local directory and records the checksum and
file inventory. The release operator publishes the identical directory under
`PREFIX/<artifact_digest>/`, keeping each relative filename. This allows the
same approval to identify the exact S3 content without a governance schema
change. Use immutable digest prefixes and retain objects needed for rollback.
S3 publication is a release-operator action. The inference role reads artifacts.

The adapter bounds the file count and streaming download size, verifies the
approved package checksum before publishing a local cache directory, and
verifies cached contents before loading. Configure a restricted, dedicated
`INFERENCE_S3_CACHE_ROOT` and monitor its disk usage. Prune cached directories
when the service is stopped, since runtimes can hold package references.
When no S3 root is configured, the existing private mount profile is used.
