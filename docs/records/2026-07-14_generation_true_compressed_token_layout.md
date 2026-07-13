# True compressed-token generation layout

Date: 2026-07-14

Branch: `scale/generative-system`

## Audit finding

The first generation-scale K8 configuration emitted eight full-resolution
`64 x 256 x 256` tensors and applied channel masks only inside `S_k`. Its first
effective field (`8 x 256 x 256`) already contained more scalars than the target
RGB epsilon field (`3 x 256 x 256`), and masked channels could still participate
in predictor feedback. That implementation remained useful as ordered,
restricted dense-component factorization evidence, but it did not satisfy the
project's stronger compressed denoising-token definition.

The completed pinned run is therefore classified as legacy evidence. It is not
permitted to authorize the full ImageNet-256 topology.

## Production layout

Authoritative scaling and full configs now use:

```text
token spatial strides:  [16, 16, 8, 8, 4, 4, 2, 1]
token spatial sizes:    [16, 16, 32, 32, 64, 64, 128, 256]
token channels:         [4, 4, 8, 8, 8, 8, 4, 2]
token scalar counts:    [1024, 1024, 8192, 8192, 32768, 32768, 65536, 131072]
dense epsilon scalars:  196608
```

Every token is strictly smaller than the dense target and token capacity is
nondecreasing. The full sequence contains `280,576` scalars (`1.427x` the dense
field); the claim is per-token compression, not aggregate sequence compression.

`ScalableUNetTokenPredictor` pools its final feature state before each true
variable-channel head. Only the emitted token is bilinearly restored for
next-token feedback. There are no hidden inactive channels in the authoritative
layout. `RestrictedSynthesis` receives only that token, applies its bias-free
`1x1` projection at token resolution, performs fixed bilinear upsampling, and
uses one bias-free local linear convolution. The zero-token invariant remains
exact.

## Fairness and lifecycle

The checked-in compressed scaling/full preflights report:

```text
CoFiTok:         62,837,576 parameters
dense_identity: 62,824,707 parameters
relative gap:   +0.020484%
```

The old pinned CoFiTok/dense 50K reports may validate only under the explicit
`legacy_scaling` recipe during fast-forward deployment. After deployment, the
completion pipeline must train a fresh compressed 10% CoFiTok/dense pair on the
same target revision and with bound dataset/checkpoint provenance. Only the new
pair's 10K sampling reports and promotion gate may authorize full 300K training.
Its post-evaluation verifies native sidecars with `integrity-policy=required`;
the legacy migration utility is not allowed to rewrite the new training report.
The completion audit separately binds the legacy deployment source revision,
the authoritative compressed 10% revision, and the full-training revision.

## Verification

- Core implementation commit: `35d21c3c153ea655778a8e490e7410619aa8f1d4`.
- Native-sidecar post-eval fix: `9cfecade12c9e2bb3786d3ae842f203b56ce993e`.
- Prerequisite-aware bundle from pinned `781a01444fddbf0d48a427ba58bdeed50167b5be`:
  `476,888` bytes, SHA256
  `5625218f7de1bc9bd8700745fb73b677bcb80bdf058d52f9d20dbccbb4fa05fd`.
- Full local suite: `571/571` pass.
- Production scaling config preflight: pass.
- Production full config preflight: pass.
- Isolated Linux target-revision suite: `571/571` pass, 0 failures/errors/skips.
- Target-tracked Linux runbooks: `40/40` `bash -n` pass.
- Bounded deployment conflict scan: 160 target-added paths, 0 conflicts.
- Formal worktree HEAD remained pinned and tracked-clean after verification.
- Tests cover variable token channels/shapes, fixed upsampling to image space,
  per-token compression, nondecreasing capacity, legacy recipe compatibility,
  matched parameter bounds, pipeline stage ordering, and completion provenance.

At record time the pinned legacy CoFiTok run was complete and the matched dense
run remained healthy. No remote revision or active training process was changed.

Machine-readable evidence:

```text
artifacts/reports/generation/compressed_scaling_config_preflight.json
artifacts/reports/generation/compressed_full_config_preflight.json
artifacts/reports/generation/compressed_token_linux_pytest.xml
artifacts/reports/generation/compressed_token_linux_runbook_syntax.json
artifacts/reports/generation/compressed_token_deployment_conflicts.json
```

Evidence SHA256 values, in the order above:

```text
compressed_scaling_config_preflight.json: recorded in the implementation commit
compressed_full_config_preflight.json: recorded in the implementation commit
compressed_token_linux_pytest.xml: 397a1846eb965811b1bee50d1a6082f6308b19ce688ab1b526159c1757275ff7
compressed_token_linux_runbook_syntax.json: 9c60474645c25e3b929b91b5d993a3abfc98bfc18f870f23bc30dfbe886f22e4
compressed_token_deployment_conflicts.json: 051e7da3f4d5da564d6cfdfbf9c25e032c65ef9b86251c9078ca2371ccddd7f5
```
