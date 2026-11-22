# Access control

The API validates RS256 bearer tokens against the configured OIDC issuer,
audience, and JWKS endpoint. Viewer tokens carry `inference_viewer` and can
browse registry metadata and requester-scoped read endpoints. Researchers
carry `inference_user` and can submit inference. Administrators carry
`inference_admin`, manage model caches, access cross-user history, and view
system status. Governance uses a separate service identity with
`inference_reader`.

## Upload controls

Prediction endpoints require a known request length. The HTTP layer bounds
total bytes before multipart parsing and limits concurrent uploads per
process. Image validation independently checks decoded format, size, and
dimensions. Configure corresponding upload limits and request timeouts at
the deployment ingress. Run with HTTPS and restrict administrative access
to authorized OIDC identities.
