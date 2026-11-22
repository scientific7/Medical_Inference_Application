import pytest
from medical_imaging_contracts.models import DatasetReleaseReference


def test_reference_maps_producer_v2_without_claiming_authentication():
    receipt = {"schema_version": "2.0", "dataset_id": "synthetic", "release_id": "v1",
               "release_status": "published", "partitions": {
                   split: {"uri": f"s3://example/{split}", "sha256": char * 64}
                   for split, char in zip(("train", "validation", "test"), "abc")}}
    ref = DatasetReleaseReference.from_verified_portable_v2(receipt, "s3://example/receipt", "d" * 64)
    assert ref.schema_version == "1.0"
    assert ref.manifest_sha256["validation"] == "b" * 64
    receipt["partitions"]["val"] = receipt["partitions"].pop("validation")
    with pytest.raises(ValueError):
        DatasetReleaseReference.from_verified_portable_v2(receipt, "s3://example/receipt", "d" * 64)
