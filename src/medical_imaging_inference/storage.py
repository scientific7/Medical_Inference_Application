"""S3 artifact retrieval for content-addressed, governance-approved packages."""

import contextlib
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

MAX_PACKAGE_FILES = 1024
MAX_PACKAGE_BYTES = 4 * 1024**3


def artifact_store_config():
    uri = os.getenv("INFERENCE_S3_ARTIFACT_ROOT")
    if not uri:
        return None
    root = urlsplit(uri)
    if (
        root.scheme != "s3"
        or not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", root.netloc)
        or root.query
        or root.fragment
        or not root.path.strip("/")
    ):
        raise ValueError("Invalid S3 artifact root")
    return root.netloc, root.path.strip("/")


def s3_client():
    import boto3
    from botocore.config import Config

    return boto3.client("s3", config=Config(connect_timeout=3, read_timeout=20, retries={"max_attempts": 2}))


def inventory(record):
    if record.get("state") != "approved":
        raise ValueError("Artifact requires explicit approval")
    digest = record.get("artifact_digest", "")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Invalid approved package digest")
    metadata = record.get("metadata")
    files = metadata.get("package_files") if isinstance(metadata, dict) else None
    if not isinstance(files, list) or not 1 <= len(files) <= MAX_PACKAGE_FILES:
        raise ValueError("Invalid approved package inventory")
    for filename in files:
        if (
            not isinstance(filename, str)
            or len(filename) > 512
            or not re.fullmatch(r"[A-Za-z0-9_. /-]+", filename)
            or any(part in {"", ".", ".."} for part in filename.split("/"))
        ):
            raise ValueError("Unsafe approved package filename")
    if len(set(files)) != len(files) or "model_manifest.json" not in files:
        raise ValueError("Incomplete or duplicate approved package inventory")
    return digest, files


def s3_package(record, config, verify):
    from botocore.exceptions import BotoCoreError, ClientError

    bucket, prefix = config
    digest, files = inventory(record)
    cache = Path(os.getenv("INFERENCE_S3_CACHE_ROOT", "./.local/s3-artifacts")).resolve()
    cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination = cache / digest
    if destination.is_symlink():
        raise ValueError("Unsafe S3 cache entry")
    if destination.is_dir():
        return destination, verify(destination, record)

    client = s3_client()
    remaining = MAX_PACKAGE_BYTES
    with tempfile.TemporaryDirectory(dir=cache) as directory:
        staging = Path(directory) / "package"
        staging.mkdir(mode=0o700)
        try:
            for name in files:
                obj = client.get_object(Bucket=bucket, Key=f"{prefix}/{digest}/{name}")
                path = staging / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with contextlib.closing(obj["Body"]) as body, path.open("xb") as output:
                    if obj["ContentLength"] > remaining:
                        raise ValueError("Approved package exceeds the download limit")
                    size = 0
                    for chunk in body.iter_chunks(chunk_size=1024 * 1024):
                        size += len(chunk)
                        remaining -= len(chunk)
                        if remaining < 0:
                            raise ValueError("Approved package exceeds the download limit")
                        output.write(chunk)
                    if size != obj["ContentLength"]:
                        raise ValueError("Incomplete artifact download")
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in {"404", "NoSuchKey", "NoSuchBucket", "NotFound"}:
                raise ValueError("Approved S3 artifact is missing") from exc
            if code in {"403", "AccessDenied"}:
                raise ValueError("Approved S3 artifact access denied") from exc
            raise ValueError("Approved S3 artifact could not be retrieved") from exc
        except BotoCoreError as exc:
            raise ValueError("Approved S3 artifact could not be retrieved") from exc
        manifest = verify(staging, record)
        try:
            staging.rename(destination)
        except OSError:
            if not destination.is_dir() or destination.is_symlink():
                raise
            manifest = verify(destination, record)
    return destination, manifest


def artifact_store_status():
    try:
        config = artifact_store_config()
        if config is None:
            return "local mount", "not configured"
        bucket, prefix = config
        s3_client().list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/", MaxKeys=1)
        return "s3", "reachable"
    except Exception:
        return "s3", "unavailable"
