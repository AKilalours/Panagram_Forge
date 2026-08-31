# Terraform

Phase 7/8. Target resources:

- S3 bucket matching the layout in `docs/data_spec_v1.md` section 8
- GPU training instances (spot, with checkpoint-resume making preemption survivable)
- model registry bucket with versioning and object lock
- IAM roles: one for training (read data, write checkpoints), one for serving
  (read model only). The serving role must not be able to read the raw corpus.
- budget alarm, because a forgotten multi-GPU instance is the most common way a
  personal project becomes expensive

Local development uses MinIO with the same paths, so promotion to cloud is a config
change rather than a rewrite.
