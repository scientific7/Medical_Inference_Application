# Security & PHI Protection Audit — Medical Inference Application

## 1. Authentication & Authorization (`security.py`)
- **Production:** Fail-closed OIDC RS256 JWT validation (`OIDC_ISSUER`, `OIDC_JWKS_URI`, `OIDC_AUDIENCE`) with RBAC roles (`inference_viewer`, `inference_user`, `inference_admin`).
- **Local Development:** `INFERENCE_AUTH_MODE=dev` restricted strictly to loopback hosts (`127.0.0.1`, `::1`, `localhost`, `testclient`) and forbidden when `INFERENCE_ENVIRONMENT=production`.

## 2. Input & Artifact Security
- **Upload Limits:** `UploadLimits` middleware caps single images at `20 MB` (`40M` pixels) and batches at `40 MB` (`8` images).
- **Package Integrity:** `security.verify_directory` verifies SHA-256 checksums, blocks symlinks and directory traversal, and refuses unapproved model versions.
- **PHI Protection:** Prediction records persist only SHA-256 input digests and metadata — raw medical images and patient identifiers are never written to PostgreSQL/SQLite logs or telemetry payloads. DICOM uploads strip patient-identifying tags (`PatientName`, `PatientID`, `PatientBirthDate`) before displaying metadata.
