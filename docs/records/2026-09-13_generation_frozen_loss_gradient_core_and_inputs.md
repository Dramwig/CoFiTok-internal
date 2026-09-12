# Frozen loss-gradient core and physical input catalog

## Progress and scope

The previous goal turn made progress: it archived the completed 20K classifier
diagnostic and independently checked its arithmetic. This turn implements the
next measurement core and freezes actual inputs. It does not claim the gradient
experiment has run or that support collapse is resolved.

Implementation commit: `5ac619228b051d12d87a2d790a3676d31c8d7b72`.
Tree: `693704cc227e7ac213be3686a920214556525058`.
Local branch: `analysis/generation-class-support-contingency-v1-20260910`.

New production-independent components:

```text
src/cofitok/generation/frozen_loss_gradients.py
scripts/build_frozen_loss_gradient_source_catalog.py
tests/test_generation_frozen_loss_gradients.py
tests/test_frozen_loss_gradient_source_catalog.py
```

The measurement API imports unchanged production forward/loss implementations,
uses `autograd.grad`, and has no optimizer, checkpoint loader, output writer,
sampler, or permission-granting code. A guarded executable is still required
before real checkpoint measurement.

## Measurement coverage

The core separates weighted epsilon, rollout, EMA-teacher and remaining
auxiliaries, checks that their scalar sum matches the production loss, and
measures parameter-group norms/dot products/cosines. Class embedding,
conditioning projections, output/feedback heads and shared trunk are disjoint
groups. Unused gradients, numerical zero gradients and undefined cosines are
represented distinctly. Nonfinite values fail closed.

Reversible hooks record the actual effective labels entering the class embedding
in the main, teacher and rollout passes. Each rollout output records raw-x0
saturation, exact clamp-boundary fraction and the gradient of weighted rollout
loss with respect to that epsilon prediction. The core also records the residual
between summed per-term gradients and the direct combined backward; mixed-precision
rounding means these must not be assumed identical.

Model/EMA tensor identities and version counters, preexisting `.grad` references,
module training flags, hooks and caller RNG state are protected/restored. Tests
add exact before/after tensor equality checks. No production training, model,
loss or sampling file was edited.

The protocol was clarified before execution: at frozen batch 64 the teacher
selects four images and rollout up to eight, whereas ceil-based selection in the
one-image probe selects 1/1 when active/valid. Results will explicitly describe
**one selected example**, not the original minibatch gradient. The weights and
scheduled scales are retained; this does not silently reweight the original
optimizer objective or claim a full optimizer-step causal attribution.

## Physically prepared inputs

The exact clean remote builder checkout is:

```text
/tmp/cofitok-frozen-loss-gradient-preparation-5ac6192
analysis/frozen-loss-gradient-preparation-v1-20260913
```

Incremental bundle (required prerequisite `c38291b14058746712dc0934937539a86bb3800e`):

```text
/tmp/cofitok-frozen-loss-gradients-5ac6192-from-c38291b.bundle
bytes: 42045
sha256: 003125471201b39f43bb54c806663af52d52c6136107e0dd9d0cd9b82f0454ca
advertised revision: 5ac619228b051d12d87a2d790a3676d31c8d7b72
```

Local and remote prerequisite/head/hash checks passed. The new isolated checkout
did not move any existing execution/validator checkout.

The new catalog is:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/.frozen_loss_gradient_attribution_v1.control/source_catalog.json
bytes: 32829
sha256: f6d1a1917db350a20d4d953f45b76e82050b51a7b2fe8ed0b7e2432bde5b929b
mode: 0444
status: prepared_inputs_only
execution_authorized: false
```

The builder and physical replay both returned `status=pass`; replay preserved
mtime `1789231703606575599` ns. This is a deterministic input-catalog replay,
not an independent gradient-result validation receipt.

The catalog binds the existing 20K diagnostic result/receipt, both 100K
checkpoints and sidecars, their training/sampling reports and recorded configs.
It rehashes both checkpoint payloads without deserializing them:

| Method | Bytes | SHA256 |
| --- | ---: | --- |
| CoFiTok | 1,010,937,514 | `b36a92229ba2dd021db9c7585970ddda1d17d1919ca6b0eec0362b06d4bd462e` |
| dense identity | 1,010,735,510 | `b6586cc906a9c38bbf9d592f5f6169b7a37a8d3aa485ca4879021fa840943d0b` |

The 405,484,553-byte dataset manifest at SHA
`9a2eec642f0d56162bffaafed84a41267f22abfc9feff4cf41fed9f6881173f0`
was streamed and replayed: 1,281,167 train and 50,000 validation rows, with
1,000 lexicographically ordered WNID labels. A bounded filesystem census found
16 historical `conditioning_sensitivity_manifest.json` files containing 32
distinct classes/image hashes. Those entire classes were excluded. Generated
`prefix_*`/`samples_*`, metric and cache subtrees were explicitly pruned; the
catalog states that scope rather than claiming an unbounded search of all prior
uses of validation data.

Seed 2031 and a fixed SHA256 ranking select 32 fresh classes, one image per
class. All 32 image bytes were hashed and their RGB 256x256 headers verified;
paths, labels, WNIDs and SHA256 values are permanently bound. Selected labels:

```text
406 744 822 988 188 563 268 370 381 589 53 831 551 884 175 605
325 729 556 455 120 233 106 826 825 215 669 759 688 277 217 204
```

This selection avoids reuse from the enumerated sensitivity probes. It is not a
claim that the images/classes were never seen by any historical validation use.

## Verification and remaining execution work

Local project CPU and exact clean Linux CPU-only focused suites each passed
**38 tests**. They cover dense and fixed-basis auxiliary gradients, deterministic
replay, mode/hook/RNG/weight preservation, teacher inactivity before its scheduled
start, invalid rollout timesteps, saturated loss, zero/unused gradients,
nonfinite and malformed input rejection, no silent CPU fallback for bf16, and
order-invariant fresh-class selection. The default 1,000-step cosine schedule
also matches the frozen 100K source's old formula elementwise on CPU.
`py_compile` and `git diff --check` pass. This is not a full-repository test claim.

Before GPU execution, still implement and test the bounded executable, its exact
source/runtime compatibility checks, result validator and immutable receipt;
build/replay the complete preparation and obtain a distinct exact diagnostic
authorization. The classifier-only approval cannot authorize this stage.
The catalog never loads a checkpoint or model and grants no GPU, training,
sampling, confirmation, full/300K, export, release or paper-integration permission.

The four immutable terminal-SNR arms and final result/receipt were replayed again
this turn: all metrics/checkpoints/sidecars/samples/reports remain intact and the
scientific result remains hold. The source controller is complete, not running.
GPU was idle, no CoFiTok compute process remained, and free generation storage
was 189,696,655,360 bytes. No source run, historical controller or paper artifact
was changed. This catalog and the isolated code checkout are the only new remote
preparation artifacts.
