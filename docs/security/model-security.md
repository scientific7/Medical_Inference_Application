# Model security

Inference verifies exact-version approval against governance and checks the
package digest before framework loading. Serve only exports produced and
reviewed by trusted publishers. TorchScript and SavedModel can execute model
operations and should be treated as executable artifacts. A hash proves the
package matches the approved bytes. Approval review establishes publisher trust.

Protect local package mounts and S3 prefixes against writes by the inference
identity. Use [S3 package provisioning](../integration/s3-integration.md) for
content-addressed releases.
