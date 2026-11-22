# Authentication

Send the OIDC API access token using the `Authorization: Bearer` header.
Configure the issuer, audience, JWKS URL, and optional role claim on the
backend. The workstation accepts a short-lived access token through its
password input. Obtain the token from the deployment's identity provider.

The backend uses a separate scoped token for governance approval. Keep
this token in the backend secret store. See [role permissions](../security/access-control.md).
