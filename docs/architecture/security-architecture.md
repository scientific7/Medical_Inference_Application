# Security architecture

OIDC identity establishes the user's API permissions. A scoped service
credential obtains approval from governance. Package verification binds the
runtime to approved artifact bytes. The production ingress and AWS identities
bound external access to the deployed components.

See [access control](../security/access-control.md), [model security](../security/model-security.md),
and [medical data privacy](../security/medical-data-privacy.md).
