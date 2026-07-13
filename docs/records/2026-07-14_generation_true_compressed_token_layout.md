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
- Guarded deployment waiter: `24c2154baeadca8c25b0125d89fe5ddbf4732903`.
- Prerequisite-aware bundle from pinned `781a01444fddbf0d48a427ba58bdeed50167b5be`:
  `479,615` bytes, SHA256
  `fecc6053c5108e0295e2ac4f2fb0515277b05e2a7ec34d0bdcc82ce5b0e63efc`.
- Full local suite: `572/572` pass.
- Production scaling config preflight: pass.
- Production full config preflight: pass.
- Isolated Linux target-revision suite: `572/572` pass, 0 failures/errors/skips.
- Target-tracked Linux runbooks: `41/41` `bash -n` pass.
- Bounded deployment conflict scan: 164 target-added paths, 0 conflicts.
- Formal worktree HEAD remained pinned and tracked-clean after verification.
- Tests cover variable token channels/shapes, fixed upsampling to image space,
  per-token compression, nondecreasing capacity, legacy recipe compatibility,
  matched parameter bounds, pipeline stage ordering, and completion provenance.

At record time the pinned legacy CoFiTok run was complete and the matched dense
run remained healthy. No remote revision or active training process was changed.

The final bundle and deployment helper may be staged under `/tmp` while training
continues. `wait_for_legacy_pair_and_deploy.sh` is a locked, 48-hour-bounded
one-shot waiter: it verifies the bundle/target and clean pinned worktree, polls
the authoritative monitor, requires both exact 50K reports and no remaining
legacy training/runbook process, then invokes the existing fail-closed deployer.
Monitor failure, timeout, revision drift, dirty tracked files, incomplete reports,
or a mismatched bundle stops deployment. It does not touch the formal worktree
while the legacy pair is active. The polling sleep closes the inherited lock
descriptor, so an interrupted waiter cannot leave an orphaned sleep holding the
deployment lock and blocking immediate recovery.

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
compressed_token_linux_pytest.xml: 50503c095f8f1ab897aae812e9ccf108d8892e7c3f8455f71de73b35cf178f1f
compressed_token_linux_runbook_syntax.json: 98ba0fa936922024e6447549a802a6abe69f68751272e5fdc19de2825e818cbb
compressed_token_deployment_conflicts.json: ed1b7f196cdacd5162836a0622452aa1bc510b9b3581c8bd56a1e78f888cfd8f
```
