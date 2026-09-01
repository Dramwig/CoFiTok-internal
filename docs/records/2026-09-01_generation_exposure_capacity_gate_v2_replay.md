# Exposure/capacity gate v2 replay (2026-09-01)

## Scope

This record documents a fresh Linux replay of the source-bound candidate gates
using the v2 control-plane builder. It is non-authorizing: it did not create a
stage authorization, launch training or sampling, modify the formal 100K
checkout, or change locked evidence.

## Replay identities

- Builder bundle: `/tmp/exposure-capacity-gate-builder-v2-4567a54-full.bundle`
  (SHA256 `576b0e11e83002b5e13f1cd6bee7256f36e0c593f8cb33c64b99046c0d4d7949`).
- Builder checkout: revision
  `4567a54abd6cd474625d722af8a0d2df21e24a27`, tree
  `c10a686734a4d11747a335619f0220240015a05b`, branch
  `analysis/generation-exposure-continuation-v1`, tracked clean.
- Locked source checkout: revision
  `cf0e5faa94bf4ab38d947b921935b3b765b5537a`, tree
  `6cef27723196fd363379bca2e7b85b1678ebd777`, branch
  `scale/generation-stability-quality-bridge-100k`, tracked clean.
- Preparation: `/tmp/cofitok-exposure-capacity-gate-20260831/evidence/preparation.json`,
  8,364 bytes, SHA256
  `8be2b6d8f2f85531b0d4f18ce8f28d93e76d739d0530175edaad33be51d51d8a`.

## Results

Both reports were built from the same preparation and independently validated
with the v2 validator (`source_count=10`):

| candidate | bytes | SHA256 | validator | execution ready | stage authorization |
| --- | ---: | --- | --- | --- | --- |
| exposure continuation | 5,433 | `0687e779b96ad28487d4b6bbabab7ad7ba023ba93597b7736fc3129089a033a0` | pass | false | not authorized |
| capacity qualification | 4,154 | `7430de5e7ecc4551da807b86e2055ced46c96c67e994b48c1ee8795e5db5d261` | pass | false | not authorized |

The exposure contract records `preserve_source_scheduler_horizon` with source
and effective horizon 100,000, target horizon 110,000, and explicit resume
target required. Both reports retain `terminal_status=hold`,
`generation_advantage_proven=false`, and all GPU/training/sampling,
promotion/export/release, and process-signal permissions false.

## Runtime boundary

At build time the target GPU reported 0 MiB used with no compute applications,
the candidate output roots were absent, and the source checkpoint payload,
sidecar, and `latest.json` bindings verified. No stage authorization file was
created and no remote training, sampling, promotion, or release process was
started.

