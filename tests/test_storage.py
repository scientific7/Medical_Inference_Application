import hashlib
import io
import json

import pytest

from medical_imaging_inference import storage
from medical_imaging_inference.security import verified_package

pytest.importorskip("botocore")
from botocore.exceptions import ClientError
from botocore.response import StreamingBody


@pytest.fixture
def package(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_S3_ARTIFACT_ROOT", "s3://test-bucket/approved")
    monkeypatch.setenv("INFERENCE_S3_CACHE_ROOT", str(tmp_path))
    weights = b"synthetic artifact"
    manifest = {"weights": {"path": "weights/export.ts", "sha256": hashlib.sha256(weights).hexdigest()}}
    files = {"weights/export.ts": weights, "model_manifest.json": json.dumps(manifest).encode()}
    digest = hashlib.sha256(
        "".join(f"{name}:{hashlib.sha256(files[name]).hexdigest()}\n" for name in sorted(files)).encode()
    ).hexdigest()
    record = {"state": "approved", "artifact_digest": digest, "metadata": {"package_files": list(files)}}
    return record, files, manifest


def test_download_verifies_and_reuses_package(package, monkeypatch):
    record, files, manifest = package
    calls = []

    class Client:
        def get_object(self, *, Bucket, Key):
            calls.append(Key)
            assert Bucket == "test-bucket"
            prefix = f"approved/{record['artifact_digest']}/"
            assert Key.startswith(prefix)
            data = files[Key[len(prefix) :]]
            return {"Body": StreamingBody(io.BytesIO(data), len(data)), "ContentLength": len(data)}

    monkeypatch.setattr(storage, "s3_client", Client)
    path, result = verified_package(record)
    assert result == manifest
    assert verified_package(record)[0] == path
    assert len(calls) == 2
    (path / "weights/export.ts").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verified_package(record)


@pytest.mark.parametrize(("code", "message"), [("AccessDenied", "access denied"), ("NoSuchKey", "missing")])
def test_s3_rejections_clean_partial_download(package, monkeypatch, tmp_path, code, message):
    record, _, _ = package

    class Client:
        def get_object(self, **_):
            raise ClientError({"Error": {"Code": code}}, "GetObject")

    monkeypatch.setattr(storage, "s3_client", Client)
    with pytest.raises(ValueError, match=message):
        verified_package(record)
    assert list(tmp_path.iterdir()) == []


def test_inventory_rejects_path_escape(package):
    record, _, _ = package
    record["metadata"]["package_files"].append("../outside")
    with pytest.raises(ValueError, match="Unsafe"):
        verified_package(record)
