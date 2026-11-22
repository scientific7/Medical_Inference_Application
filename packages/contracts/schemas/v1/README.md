# Contract schemas v1

The Pydantic models in the installable `medical_imaging_contracts` package are the executable wire contract. JSON Schema can be generated from each model with `model_json_schema()`; schemas are versioned by package major version and this directory. Additive optional fields are compatible; changing/removing required fields requires a new major contract version.

Dataset release references pin the exact receipt digest and per-partition manifest digests. Current Dataset Pipeline receipts are authenticated HMAC-SHA256; consumers must additionally pin the receipt SHA-256 from a trusted channel. Never treat a digest alone as proof of publisher identity.
