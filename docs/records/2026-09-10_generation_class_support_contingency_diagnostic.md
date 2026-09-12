# Generation class-support contingency diagnostic (2026-09-10)

> Historical preparation record. The diagnostic was subsequently authorized and
> executed once on September 12, with its existing validation receipt physically
> replayed on September 13. See
> [the execution and interpretation record](2026-09-13_generation_class_support_contingency_result.md).
> Statements below about absent execution artifacts describe the preparation
> checkpoint, not the current state. No frozen source was rewritten.

## Scope and boundary

This record documents a source-bound, non-authorizing CPU diagnostic prepared
for the completed terminal-SNR generation screen. It analyzes the frozen
10K-sample trees from the 100K EMA checkpoints for possible class-support failure modes
(ignored, misaligned, permuted, or shared unconditional support), but it does
not create samples, load a classifier on the server, retrain either method,
change a checkpoint, or replace the terminal-SNR result.

The diagnostic is deliberately contingent. It cannot authorize or imply a
confirmation run, large-capacity readiness, full training, 300K training,
promotion, export, release, or paper integration. A source-bound,
non-authorizing preparation was subsequently created and physically replayed.
No stage approval, execution authorization, classifier inference, diagnostic
result, or validation receipt has been created.

The implementation was developed in the isolated worktree `C:/qbcsc` on
branch `analysis/generation-class-support-contingency-v1-20260910`, based on
revision `94f7dd1a5833f76eda66ef8880a704e9730f4602`. The owned files are:

```text
src/cofitok/generation_class_support_contingency.py
scripts/build_generation_class_support_contingency_preparation.py
scripts/build_generation_class_support_contingency_execution_authorization.py
scripts/evaluate_generation_class_support_contingency.py
scripts/validate_generation_class_support_contingency.py
tests/test_generation_class_support_contingency.py
```

## Frozen source contract

The code binds the fixed torchvision ResNet-50 ImageNet-1K V2 classifier
(`102,540,417` bytes, SHA256
`11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`) and the
authoritative 100K sampling revision
`cf0e5faa94bf4ab38d947b921935b3b765b5537a` on
`scale/generation-stability-quality-bridge-100k`. The diagnostic contract is
exactly 10,000 samples per method, ImageNet-256 RGB `[3, 256, 256]`, EMA
DDIM-100, seed 0, CFG 1.5, the recorded balanced-modulo class schedule,
and the recorded per-global-index random-stream and sample-digest framing.
It treats the frozen CoFiTok K8 and dense-identity sample trees, checkpoints,
sampling reports, manifests, progress reports, generation metrics, and class
fidelity reports as immutable source evidence.

The causal decision is fixed to
`no_defensible_shared_intervention_selected`. The valid source-bound causal
SHA256 is
`6b0887dec10d6908550bee8e90c3a6e242dbddb3cb62c97935eac841dd262bc0`; the
independent validation binding is
`0043680c6109de3adffae743517ef87b4efd81f3e87f6d540a144aa804354292`.

## Immutable remote preparation

The non-authorizing preparation is stored outside the frozen sample trees and
outside the immutable terminal-SNR screen:

```text
path:   /root/autodl-tmp/CoFiTok/checkpoints/generation/.class_support_contingency_v1.control/preparation.json
bytes:  19,469
sha256: 4a1aebfcb2ee5b32586d97a2a1d39ec619048fcd99840fef7dfce9033dabc2bc
```

It was built from the clean preparation checkout
`/tmp/cofitok-class-support-contingency-preparation-c38291b` at revision
`c38291b14058746712dc0934937539a86bb3800e` and tree
`ae5d00d450cee99777ed1f663e92d179511204ed`. The preparation binds the fixed
classifier above and physically verified frozen sample trees with these exact
identities:

| method | images | sample-set SHA256 |
| --- | ---: | --- |
| CoFiTok K8 | 10,000 | `7f57ee4a874667b17085df203503443440a84124719a53da31db726a1efaff98` |
| dense identity | 10,000 | `3fc905a55dc368ddf268278c04963c5933b3e783f4cabd872762135b6345bdaf` |

The same exact checkout replayed the preparation from every physical source.
The latest replay returned `status=pass`,
`scientific_status=diagnostic_not_executed`,
`classifier_inference_performed=false`, and `execution_authorized=false`.
The intended result root remains absent:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/class_support_contingency_v1
```

The preparation itself fixes every mutation and downstream permission to
false. In particular, it is not a stage approval or execution authorization.
Running the frozen 20,000-image classifier diagnostic requires a distinct
user-created approval bound to this preparation identity and to the exact
evaluator Git identity. That approval is limited to existing-image classifier
inference; it cannot authorize sampling, retraining, confirmation, full
training, 300K training, promotion, export, release, or paper integration.

## Hardening included

The contingency module and its four scripts enforce:

- exact preparation, approval, authorization, result, and validation schemas;
- physical numbered-PNG verification, per-image SHA256 replay, regular-file
  and symlink rejection, and directory/file stat checks before and after
  hashing;
- exact frozen sampling-schema comparison, including 100 DDIM timesteps,
  random-stream formula, digest framing, typed inference fields, and the
  authoritative report layout;
- strict integer and finite-number validation, including exact binomial
  handling at probabilities zero and one;
- AMI and permutation nulls, cyclic offsets, direct and held-out two-fold
  one-to-one mappings, direct-baseline lifts, histogram overlap/JSD, and
  duplicate accounting;
- evaluator and validator rehashes of every bound source, Git identity, frozen
  tree, and prediction file; and
- elapsed-time stamping after deterministic analysis so a caller-supplied
  classifier interval is included without mutating an immutable result.

## Local verification

Using the project `uv` environment with the development and quality extras:

```text
focused contingency tests: 13 passed
compileall (src, scripts, tests): passed
repository CPU suite with CUDA hidden: 1263 passed, 12 skipped
```

The full suite completed in 309.36 seconds. The later preparation step wrote
only the immutable JSON preparation described above. It performed no
classifier inference, sampling, training, checkpoint mutation, controller
mutation, or GPU work.

## Live authoritative screen audit

The active terminal-SNR controller remains completed with a terminal hold:

```text
controller_status.json SHA256: dfba8a85c6aa0b530e27874c5a0187f18ace3ac2624ab0cfedd92b8f4dd45e14
controller.log SHA256:        2167f99c0c899c7e79896dc2924849e9a5bd8638e9c985777f13f97edaca084d
execution revision/tree:      89bcd9adb2a1e4625a9cd48dc2be82cbee8c6430 /
                              46efd20cff489bccd799bb13c4155a0cc79e7649
```

All four terminal-SNR arms were physically replayed from their immutable
reports and source paths. Each has 201 finite, strictly increasing metric
rows through step 10,000, with `samples_seen = step * 64`; each 5K and 10K
checkpoint was rehashed against its integrity sidecar, `latest.json`, and
report. Each 1,000-image PNG tree passed physical verification and digest
replay:

| arm | checkpoint bytes | 10K checkpoint SHA256 | sample-set SHA256 |
| --- | ---: | --- | --- |
| control_cofitok | 1,010,933,866 | `d40b6d04d7d50eeadfd8db431c8cf4ff0407173cd9d1f48ab852701270d9b15e` | `acea3aa8f6872fead501401e95c041f5e2d7e033ac9732be031ebdefcba1f0f2` |
| control_dense_identity | 1,010,732,310 | `de97b58510966473d45c4dc8959d894d63dd4735bbd40441d375a32dace60537` | `7bfbd65544fb008cabfceaa30b228fdd85dea61a3cb40bf2df0de525b2fa8c6e` |
| endpoint0975_cofitok | 1,010,933,866 | `2f39315e1bc3aed3f29696034b4d6455a74d1e25b63027179cd945bdf715aef7` | `5488032518e1396dcc74444102370d86bcf52b3dc6b12d762f6c0303887fa28b` |
| endpoint0975_dense_identity | 1,010,732,310 | `c67a57acd439a4c87ec1e12573d6fb7aca0808b6f0fcafd4192aebf9498e2b5b` | `649c35dcf38058abaa8d4aa2a6070f9838a2ff8f35b2dc0ed4cc7a31b1eb5032` |

The exact clean validator checkout
`/tmp/cofitok-terminal-snr-confirmation-rehearsal-a5199e6-v1` at
`a5199e6eea63817fe4eee59ac187766748d6e772` (tree
`2bbe3a1fdc39180f0f5d2f0fd144cebe25828296`) replayed all four arm
validations and the result. The existing adjacent receipt remains unchanged:

```text
terminal_snr_screen_result.json:            b740c8b21aabf640c156aea076058c73350d26c110e2cc967576a26492a52dea
terminal_snr_screen_result.validation.json: 3be043d852bd61d6e79005dc9d14f77113f1250bef967ca4e2a8ac74bc715f95
```

The launch receipt's immutable physical-source contract also replayed
successfully. Its builder's `training_state_absent_at_launch` precondition is
necessarily false after a run has produced output, so no post-run rebuild was
attempted and no evidence was altered.

The scientific result is still a hold, not a pass. Both
`cofitok.relative_fid_improvement` and
`dense_identity.relative_fid_improvement` fail the declared `>= 0.05`
threshold. The result therefore keeps `generation_advantage_proven=false` and
all confirmation, full-training, 300K, promotion, export, release, and paper
integration permissions false. The latest recheck found the GPU idle at
`0 MiB` and `0%` utilization, with no GPU compute process. The authoritative
`/root/autodl-tmp` free space was 189,708,275,712 bytes. The historical
exposure controller and the immutable terminal-SNR controller, evidence, and
result remain untouched.
