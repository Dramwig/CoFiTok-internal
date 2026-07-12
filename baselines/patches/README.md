# Baseline Patches

Store small patches for external baseline repositories here, grouped by alias:

```text
baselines/patches/<alias>/
```

Prefer patch files over editing remote clones in place without provenance. Each
patch record should include the upstream commit hash, reason, command used to
apply the patch, and whether it changes model behavior or only paths/logging.
