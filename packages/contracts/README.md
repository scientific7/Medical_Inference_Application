# Medical Imaging Platform Contracts

Versioned, framework-neutral Pydantic 2 wire models and generated JSON Schema for Python 3.10 services. This package defines the integration boundary; repositories must not import each other's internal modules.

## Contracts v1

`schemas/v1/` contains checked-in JSON Schema for dataset release references, dataset manifest samples, experiments, model artifacts, model registry records, deployment packages, inference requests/results, and lifecycle events. The Pydantic models in `src/medical_imaging_contracts/models.py` are authoritative. Regenerate and verify schemas with:

```bash
python3.10 -m pip install -e . pytest==8.4.2
python3.10 scripts/export_json_schemas.py
python3.10 -m pytest -q
```

The package version is `1.0.0`. Consumers pin an immutable Git commit from
`scientific7/medical-imaging-platform-contracts` until a package index is used.
The local directory name `Medical_Inference_Application` is historical: this
checkout owns contracts, while inference lives in `medical-imaging-inference-platform`.
Additive optional fields are preferred; incompatible changes require a new schema
major version and migration guidance. Schema validation does not authenticate a
publisher or approve a model.

Dataset Pipeline portable receipt v2 and these wire models v1 have separate
versions. `DatasetReleaseReference.from_verified_portable_v2()` maps an already
authenticated producer receipt into experiment lineage; it is not a verifier.

## Local research integration

The [synthetic acceptance suite](integration/README.md) exercises the actual
publisher, reader, training engine, governance API and inference API across four
checkouts. PyTorch and TensorFlow classification both train and predict. All
three task manifests are tested, including detection negatives. This needs no
institutional SSO and creates no cloud jobs. Synthetic identities are confined
to the test process; hosted authentication remains deployment configuration.

```bash
/path/to/python3.10 scripts/test_platform.py --workspace /path/to/checkouts
```
