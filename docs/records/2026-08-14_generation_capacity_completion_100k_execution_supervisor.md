# Bounded matched 250M step-100K completion execution supervisor

Date: 2026-08-14

## Purpose and boundary

This stage closes the automatic execution gap after the source-replayed
step-50K result. It can resume only the exact matched base-256 CoFiTok and
dense-identity trajectories from their selected step-50K checkpoints to the
already configured step-100K horizon. It cannot start fresh training, continue
past step 100K, authorize full 300K, promote a model, make a formal generation
claim, or authorize release.

The supervisor consumes the CPU-only decision waiter at revision `7c3a1b8`.
If the decision is a hold, the supervisor records `not_selected` and exits. If
the exact completion recommendation is selected, it waits for five consecutive
GPU-idle polls, rejects duplicate output-root processes, and starts one owned
process group. Unrelated GPU processes are observed but never signaled.

## Source preservation and exact resume

The capacity-probe configs use rolling checkpoint retention and protect the
original step-10K checkpoint, not step 50K. Before any step-100K resume, the
runbook therefore creates same-filesystem hard links for both selected step-50K
checkpoints and integrity sidecars under:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_probe_250m_10k_v1/reports/capacity_completion_100k/source_checkpoints/
```

The archive verifies the decision-bound bytes and SHA256, checkpoint integrity,
step number, and same-file identity. If rolling retention later unlinks the
original filename, retry and final validation restore that filename from the
hard link without rewriting checkpoint bytes.

Training uses the existing base-256 100K configs and the original training
checkout at revision `3a7dc9d`. For each method it calculates the remaining
delta from `latest.json` and calls `train_generation.py --resume auto
--stop-after-steps <delta>`. The completion validator requires:

- exact clean Git, resolved config, dataset, runtime environment, and parameter
  identities;
- a canonical strictly increasing metric history through step 100K with all
  100 scheduled validation events;
- exact effective batch 64 and 6,400,000 images seen;
- preserved step-50K archive plus final step-100K checkpoint and integrity;
- no later metric row or checkpoint and no authorization for more training.

## Terminal evidence

After both trainings complete, the runbook builds a matched 2,048-sample
DDIM-50 step-100K trend milestone. It then produces, per method:

- 10,000 balanced-modulo ImageNet-1K samples;
- EMA, bf16, DDIM-100, CFG 1.5, guidance rescale 0;
- FID, Inception Score, precision, recall, and class fidelity;
- a 256-image checkpoint/mechanism evaluation at timestep 500, with four
  random orders for CoFiTok.

Completion is explicitly non-authorizing. The final execution status requires
a new source-compatible result before any further decision.

## Exact implementation and deployment

```text
revision: 892c1d63db872f1135d41adaf0a25ee7388999c7
tree: a8ffccf87ff78e091e3829dca72e20d35b74b033
remote branch: scale/generation-capacity-completion-100k-execution-v1
checkout: /root/autodl-tmp/CoFiTok/checkouts/capacity-completion-100k-892c1d6/CoFiTok-internal
```

Prerequisite-aware bundle:

```text
path: /tmp/cofitok-capacity-completion-892c1d6.bundle
bytes: 41,855
sha256: e87e456fe580923d9500e6c3136bd72098d89b4108a671550be8f101e49ef700
advertised head: 892c1d63db872f1135d41adaf0a25ee7388999c7
prerequisite: 7c3a1b8a06fa4dab8733c808b284fbe86cfeaccb
```

The prerequisite was verified in the exact decision checkout before fetch. The
new checkout was created independently and remained clean.

Key deployed artifacts:

```text
execution runbook: 28,490 bytes
execution runbook sha256: 781c43fdb7bced3b5511e5a839ba8f90f13c2de7820a1718e4dd9f6c0f5775fc
supervisor runbook: 4,651 bytes
supervisor runbook sha256: e4237fb35ce256dcde77027ac9002fe76e2f8b027ac6a3c3446f7585d7c61013
supervisor script: 37,079 bytes
supervisor script sha256: 6a31eae2a24f94e535b36592b86ebe211f87c6a22f2a43e88708d1bdb1e85ec1
```

## Validation

Windows project environment:

```text
new execution/training/archive/supervisor/runbook tests: 25 passed
capacity-family regression: 149 passed
formal runbook CLI contract: passed
direct runbook-PYTHONPATH imports: passed
Python compileall: passed
git diff --check: passed
```

Linux exact checkout:

```text
capacity-family plus formal runbook CLI contract: 150 passed
eleven changed Python entrypoints/modules: py_compile passed
both new runbooks: bash -n passed
tracked status count after validation: 0
```

## Live supervisor

```text
PID: 218918
nice: 19
status: waiting
detail: waiting_for_source_replayed_capacity_completion_decision
status: /root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_probe_250m_10k_v1/reports/capacity_completion_100k_supervisor_status.json
log: /root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_probe_250m_10k_v1/reports/capacity_completion_100k_supervisor_892c1d6.log
deployment receipt: /root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_probe_250m_10k_v1/reports/capacity_completion_100k_supervisor_deployment_receipt.json
deployment receipt bytes: 8,156
deployment receipt sha256: 474446c32b8ebfd8cdbd6c3033622186220530a210c276f763cf8ec7dc9955e4
```

The initial immutable status snapshot is 1,519 bytes with SHA256
`83a21ed1fcf6eff37750b6876da0380338ed325ba7e6a5306049e758fcf52fb0`.

During deployment, the first PID pointer contained the literal ASCII suffix
`n` because of launch-shell escaping. The live process and status JSON both had
the correct PID and were not restarted. The original 7-byte PID pointer and its
first receipt were preserved as
`capacity_completion_100k_supervisor_deployment_receipt_invalid_pidfile.json`
(SHA256
`c1893935c0cdf95e018f50d44c03d286e593b823453573249d227a1af5bf6bbf`).
The canonical PID pointer was atomically corrected to six bytes, and the final
deployment receipt binds both the corrected pointer and the preserved invalid
receipt.

At deployment, FieldScope PID `910099` remained the only GPU process at 2,256
MiB. No CoFiTok GPU child was launched. The formal checkout remained at
`1ebcc15210e63a776a2ba448481cbd8bb94a4066`, with porcelain count `87` and
porcelain SHA256
`a7e1a2daf19f77e5d92efaa09b2768f36f5df2e989475480f7d22e54b46e9497`.

## Automatic behavior from here

The supervisor remains CPU-only until the step-50K execution completes and the
decision waiter emits a source-replayed terminal decision. A supported decision
still does not bypass resource safety: the supervisor waits for five consecutive
idle polls and no relevant output-root process. It retries only bounded training
or evaluation stages and launch races, and any stall termination targets only
the process group it created. It stops after verified step-100K terminal
evidence and does not continue to full 300K.
