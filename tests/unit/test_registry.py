from forge.registry.model_registry import validate_entry


def test_weights_alone_are_not_a_model_version():
    missing = validate_entry({"version": "v0.1", "weights_uri": "s3://..."})
    assert "dataset_version" in missing and "code_commit" in missing
