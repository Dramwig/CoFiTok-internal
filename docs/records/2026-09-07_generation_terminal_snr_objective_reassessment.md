# Terminal-SNR objective reassessment (2026-09-07)

## Scope

This change selects one bounded matched intervention after the completed
capacity screen held. The intervention changes only
`diffusion.cosine_endpoint_fraction` from `1.0` to `0.975`, identically for
CoFiTok and dense identity. Both methods retain epsilon prediction, the same
base-128 backbone, seed, optimization, runtime, data, losses, and fresh 10K
training boundary. It does not authorize training, sampling, evaluation,
confirmation, full training, 300K training, promotion, export, release, or
process signals.

Velocity prediction was rejected because converting velocity back to epsilon
requires an input-dependent `x_t` path outside the restricted synthesis
components. Direct x0 prediction was rejected because it changes the token
target away from compressed negative-noise components. The chosen endpoint
therefore preserves the CoFiTok method boundary: token components still sum
directly to dense epsilon prediction.

## Numerical discriminator

The implementation reproduces the training cosine schedule in float64,
including its beta clamps. At the default endpoint, the terminal
`sqrt(alpha_bar)` is `4.927305075733178e-05`, so the epsilon-to-x0 condition
factor is `20295.069711128064`. At endpoint `0.975`, the values are
`0.03894328910581123` and `25.67836520646576`. The condition factor is reduced
by `790.3567671830529` times. This is a selection rationale, not evidence that
terminal conditioning is the causal root of the prior quality hold.

Existing matched start-975 sampling diagnostics improved FID directionally
for both methods, but neither passed the complete sampling-only contract.
Those results may select a fresh training endpoint and may not be reused as a
sampler fix or promotion result.

## Implementation identity

The implementation is commit
`b68a5cc55d23c5e689c8721dce341009b60dbf8d`, tree
`a3ffa50176a6ad8a0c5f5ff22a6160e2e676b6e1`, on branch
`analysis/generation-terminal-snr-endpoint-screen-v1-20260907`. The clean
remote rehearsal checkout is `/tmp/cofitok-terminal-snr-rehearsal-b68a5cc`.

The configuration parser adds `cosine_endpoint_fraction` with default `1.0`.
The default schedule remains elementwise backward compatible. Non-default
values are accepted only for cosine schedules and must be finite in `(0, 1]`.
Two endpoint-0.975 configs and source-bound build/validation CLIs were added.
Tamper tests cover every source identity, Git provenance, config drift,
authorization escalation, and invalid schedule value.

The named-branch CUDA-disabled Linux suite reported `1232` tests, `0`
failures, `0` errors, and `6` skips in `214.827` seconds. Its JUnit SHA256 is
`16c50959561d3fac7bb9700c328c72f8741042a58cd7cd3f41f8ffdd96f412a4`.
Compilation, CLI imports, and all `110/110` tracked runbook `bash -n` checks
passed. The two failures from the first detached-HEAD attempt were provenance
checks working as designed; the exact named branch passed without weakening
them.

## Immutable reassessment evidence

The source-bound decision is:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/capacity_qualification_v1/terminal_snr_objective_reassessment.json
bytes=17364
sha256=646e47f4efab523ae3d27e1d341a48ac07c10e6c16e5fdbba912cb4e96be67af
```

It selects a fresh same-revision four-arm screen:

```text
CoFiTok endpoint 1.0
dense identity endpoint 1.0
CoFiTok endpoint 0.975
dense identity endpoint 0.975
```

Each arm is bounded to 10K optimizer steps and 640K images, followed by 1K
EMA DDIM-100 samples at CFG 1.5. A passing result can only prepare a separately
authorized frozen-checkpoint 10K-per-arm confirmation.

The exact validator physically replayed all seven source groups and created:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/capacity_qualification_v1/terminal_snr_objective_reassessment.validation.json
bytes=11625
sha256=76ce4f3bba0f0e3abfa3f06f0810c072c3773507e4106f8cd2ae6eecffc10b1a
```

The receipt binds decision and validator Git to the named clean `b68a5cc`
checkout. A second physical replay returned the same identities and left the
receipt mtime unchanged at Unix second `1788788880`. Both artifacts retain
`generation_advantage_proven=false`,
`scientific_status=bounded_screen_selected_not_executed`, and all execution,
confirmation, full-training, 300K, promotion, export, and release permissions
false.

## Live source audit

The exposure execution remains clean at
`5c23141a24a3385101ed6081c1ed656aad955a99`, tree
`ffe66d2d2fc666884a21856bbc84a23fe667f88b`. Both methods have 201 finite,
strictly increasing metric rows from step 100001 through 110000 and exact
`samples_seen=step*64` accounting. Their final reports are complete at 110K,
embed byte-identical `latest.json` objects, and their sidecars equal
`latest.json` without the sidecar pointer. Physical checkpoint hashes remain:

- CoFiTok: `994fc9ceae8cd24a67d9f35d0c4f565b74cee55b231b8d823c5908518e7620c5`.
- dense identity: `5a5bfc1d794dabfa96ee27ee9c092a748b22fa91245c9f6ba1e2741053acf28d`.

All five exposure authorization-chain files were present and matched their
bound hashes. The runtime project files also matched their recorded hashes.
The controller's historical failure remains untouched and is still the known
old runtime-SHA comparison error, not a training or checkpoint failure.

At final capture the RTX PRO 6000 reported 0 MiB used, no compute PID was
present, and no matching CoFiTok trainer, sampler, evaluator, or controller was
running. `/root/autodl-tmp` had `711438217216` free bytes. No process was
signaled and no GPU stage was launched.

## Replacement bundle

The correctly named incremental bundle is:

```text
/tmp/cofitok-generation-terminal-snr-b68a5cc-from-b15d979.bundle
bytes=40134
sha256=bd117956d231e3c85119a07c59f8dbf9c8398c4be29407ee6b68b33fa931c357
requires=b15d979d8c03d50f29d1e6e460d3ce3988152fbe
advertises=b68a5cc55d23c5e689c8721dce341009b60dbf8d refs/heads/analysis/generation-terminal-snr-endpoint-screen-v1-20260907
```

Local and remote `git bundle verify` passed. The older 40,135-byte bundle is
retained as historical output but is not an acceptable deployment bundle
because it advertises the former branch name.

## Remaining gate

The reassessment permits only construction of a separate source-bound endpoint
screen preparation. Before GPU work, that implementation must provide exact
pair validations, an explicit user-goal-bound stage authorization, execution
authorization, clean execution checkout, runtime selection, storage evidence,
an idle live snapshot, and an immutable launch receipt. Any failed or stale
source keeps the stage on hold. Full training and 300K remain unauthorized.
