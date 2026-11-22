# Training package integration

Training publishes a package directory containing a manifest, checksum,
preprocessing and class mappings, evaluation summary, model card, and model
files. Governance verifies the package and returns the approved digest and
inventory. Inference uses a checked TorchScript or TensorFlow SavedModel
export. Review the [integration profile](../INTEGRATION.md) for validated
preprocessing and evaluation fields.
