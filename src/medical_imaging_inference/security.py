"""OIDC bearer validation and bounded package verification."""

from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .storage import artifact_store_config, s3_package

bearer = HTTPBearer(auto_error=False)
VALID_ROLES = frozenset({"inference_viewer", "inference_user", "inference_admin"})
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@lru_cache(maxsize=8)
def _jwks_client(jwks_uri: str):
    import jwt

    return jwt.PyJWKClient(jwks_uri, cache_keys=True, lifespan=300)


def authorize(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> dict:
    auth_mode = os.getenv("INFERENCE_AUTH_MODE", "oidc").strip().lower()
    environment = os.getenv("INFERENCE_ENVIRONMENT", "development").strip().lower()
    if auth_mode == "dev":
        if environment == "production":
            raise HTTPException(503, "Development auth mode is forbidden in production")
        client_host = request.client.host if request.client else ""
        if client_host not in LOOPBACK_HOSTS:
            raise HTTPException(403, "Development auth mode is restricted to loopback")
        subject = (
            request.headers.get("X-Dev-User")
            or request.headers.get("X-Research-Actor")
            or "local-developer"
        ).strip()
        if not subject:
            raise HTTPException(401, "Development subject must not be empty")
        raw_roles = request.headers.get("X-Dev-Roles", "inference_user,inference_viewer,inference_admin")
        roles = [role.strip() for role in re.split(r"[\s,]+", raw_roles) if role.strip()]
        if not (VALID_ROLES & set(roles)):
            raise HTTPException(403, "Inference role is required")
        return {"subject": subject, "roles": roles}
    if auth_mode != "oidc":
        raise HTTPException(503, "Unsupported inference authentication mode")

    issuer, jwks_uri, audience = (
        os.getenv("OIDC_ISSUER", ""),
        os.getenv("OIDC_JWKS_URI", ""),
        os.getenv("OIDC_AUDIENCE", ""),
    )
    if not all((issuer, jwks_uri, audience)):
        raise HTTPException(503, "Inference OIDC configuration is required")
    if credentials is None:
        raise HTTPException(401, "Bearer token required", headers={"WWW-Authenticate": "Bearer"})
    try:
        import jwt

        signing_key = _jwks_client(jwks_uri).get_signing_key_from_jwt(credentials.credentials).key
        claims = jwt.decode(
            credentials.credentials, signing_key, algorithms=["RS256"], issuer=issuer, audience=audience
        )
    except Exception as exc:
        raise HTTPException(401, "Invalid SSO access token") from exc
    role_claim = os.getenv("OIDC_ROLES_CLAIM", "roles")
    roles = claims.get(role_claim, claims.get("groups", []))
    if isinstance(roles, str):
        roles = roles.split()
    if not isinstance(roles, list) or not all(isinstance(role, str) for role in roles):
        raise HTTPException(403, "Invalid inference role claim")
    if not (VALID_ROLES & set(roles)):
        raise HTTPException(403, "Inference role is required")
    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject:
        raise HTTPException(401, "SSO token does not identify a subject")
    return {"subject": subject, "roles": roles}


def authorize_researcher(user: dict = Depends(authorize)) -> dict:
    if not ({"inference_user", "inference_admin"} & set(user["roles"])):
        raise HTTPException(403, "Inference researcher role is required")
    return user


def verified_package(record: dict) -> tuple[Path, dict]:
    config = artifact_store_config()
    if config:
        return s3_package(record, config, verify_directory)
    root = Path(os.getenv("INFERENCE_ARTIFACT_ROOT", "./artifacts")).resolve()
    package = Path(record.get("artifact_path", "")).resolve()
    if root not in package.parents or not package.is_dir():
        raise ValueError("Approved artifact is unavailable in the configured private package mount")
    return package, verify_directory(package, record)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_directory(package: Path, record: dict) -> dict:
    if record.get("state") != "approved":
        raise ValueError("Artifact requires explicit approval")
    expected_digest = record.get("artifact_digest")
    if not isinstance(expected_digest, str) or not SHA256_RE.fullmatch(expected_digest):
        raise ValueError("Invalid approved package digest")
    if any(path.is_symlink() for path in package.rglob("*")):
        raise ValueError("Symbolic links are not allowed in approved packages")
    manifest_path = package / "model_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Approved package is missing its manifest")
    if record.get("manifest_digest") and file_digest(manifest_path) != record["manifest_digest"]:
        raise ValueError("Approved manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Approved package manifest must be a JSON object")
    if (
        manifest.get("model_id") is not None
        and record.get("model_id") is not None
        and manifest["model_id"] != record["model_id"]
    ):
        raise ValueError("Manifest model_id does not match registry record")
    if (
        manifest.get("version") is not None
        and record.get("version") is not None
        and str(manifest["version"]) != str(record["version"])
    ):
        raise ValueError("Manifest version does not match registry record")
    for field in ("task", "architecture", "framework", "preprocessing", "class_mapping"):
        approved = record.get("metadata", {}).get(field) if isinstance(record.get("metadata"), dict) else None
        if approved is not None and manifest.get(field) != approved:
            raise ValueError("Manifest metadata differs from approval")
    ref = manifest.get("weights")
    if not isinstance(ref, dict):
        raise ValueError("Approved package manifest is missing weights metadata")
    weights_rel = ref.get("path")
    expected_weights_sha = ref.get("sha256")
    if (
        not isinstance(weights_rel, str)
        or not weights_rel
        or not isinstance(expected_weights_sha, str)
        or not SHA256_RE.fullmatch(expected_weights_sha)
    ):
        raise ValueError("Approved weights metadata is invalid")
    weights = (package / weights_rel).resolve()
    if package not in weights.parents or not weights.is_file():
        raise ValueError("Approved weights path is invalid")
    digest = file_digest(weights)
    if digest != expected_weights_sha:
        raise ValueError("Approved model weight checksum mismatch")
    filenames = sorted(item.relative_to(package).as_posix() for item in package.rglob("*") if item.is_file())
    package_digest = hashlib.sha256(
        "".join(f"{name}:{file_digest(package / name)}\n" for name in filenames).encode()
    ).hexdigest()
    if expected_digest != package_digest:
        raise ValueError("Approved package digest differs from registry record")
    return manifest
