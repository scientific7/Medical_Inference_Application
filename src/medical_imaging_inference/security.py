"""OIDC bearer validation and bounded package verification."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .storage import artifact_store_config, s3_package

bearer = HTTPBearer(auto_error=False)


def authorize(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict:
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

        signing_key = jwt.PyJWKClient(jwks_uri).get_signing_key_from_jwt(credentials.credentials).key
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
    if not ({"inference_viewer", "inference_user", "inference_admin"} & set(roles)):
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
    if any(path.is_symlink() for path in package.rglob("*")):
        raise ValueError("Symbolic links are not allowed in approved packages")
    manifest_path = package / "model_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Approved package is missing its manifest")
    if record.get("manifest_digest") and file_digest(manifest_path) != record["manifest_digest"]:
        raise ValueError("Approved manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for field in ("model_id", "version"):
        if field in manifest and manifest[field] != record.get(field):
            raise ValueError("Manifest model identity differs from approval")
    for field in ("task", "architecture", "framework", "preprocessing", "class_mapping"):
        approved = record.get("metadata", {}).get(field)
        if approved is not None and manifest.get(field) != approved:
            raise ValueError("Manifest metadata differs from approval")
    ref = manifest.get("weights", {})
    weights = (package / ref.get("path", "")).resolve()
    if package not in weights.parents or not weights.is_file():
        raise ValueError("Approved weights path is invalid")
    digest = file_digest(weights)
    if digest != ref.get("sha256"):
        raise ValueError("Approved model weight checksum mismatch")
    filenames = sorted(item.relative_to(package).as_posix() for item in package.rglob("*") if item.is_file())
    package_digest = hashlib.sha256(
        "".join(f"{name}:{file_digest(package / name)}\n" for name in filenames).encode()
    ).hexdigest()
    if record.get("artifact_digest") != package_digest:
        raise ValueError("Approved package digest differs from registry record")
    return manifest
