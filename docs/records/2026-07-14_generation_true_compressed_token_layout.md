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
The completion audit separately binds the legacy deployment source revision,
the authoritative compressed 10% revision, and the full-training revision.

## Verification

- Full local suite: pass.
- Production scaling config preflight: pass.
- Production full config preflight: pass.
- Modified Linux runbooks: `bash -n` pass on `pro6000` via isolated `/tmp`
  copies; formal worktree HEAD remained pinned.
- Tests cover variable token channels/shapes, fixed upsampling to image space,
  per-token compression, nondecreasing capacity, legacy recipe compatibility,
  matched parameter bounds, pipeline stage ordering, and completion provenance.

At record time the pinned legacy CoFiTok run was complete and the matched dense
run remained healthy. No remote revision or active training process was changed.
