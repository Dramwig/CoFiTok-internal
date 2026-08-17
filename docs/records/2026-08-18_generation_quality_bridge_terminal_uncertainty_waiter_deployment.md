# Full-data 100K terminal matched-uncertainty waiter deployment

Date: 2026-08-18 (Asia/Shanghai)

## Decision

The previously deployed matched-uncertainty waiter is bound to the frozen 10%-data 50K sample trees. It waits for the active full-data 100K quality bridge only to obtain an idle GPU slot; it does not quantify uncertainty for the new terminal 100K sample pair.

A separate, permanently non-authorizing chain was therefore added for the full-data 100K terminal result. It builds its execution manifest only after replay-verifying the exact `quality_bridge_result.json`, binds the terminal CoFiTok and dense sample/checkpoint/real-set/evaluator identities, waits for the older uncertainty chain to become terminal, confirms five consecutive idle-GPU polls, and runs exactly one paired uncertainty audit.

The audit uses 10,000 matched samples per method, 20 blocks of 500 samples, five real folds, 10,000 bootstrap repetitions, and seed 3031. Scientific `pass` and `hold` are both valid terminal outcomes. Neither outcome authorizes training, 300K, release, or a broad generation-superiority claim.

## Locked implementation

- Revision: `e9a424b36bc5b951a9a27d7e1dc7f7afc650f706`
- Tree: `4f4cf5015ceb4be426e314bbd224778860e98d00`
- Branch: `analysis/generation-quality-bridge-terminal-uncertainty-v1`
- Subject: `Add terminal quality bridge uncertainty waiter`

The implementation adds:

- `scripts/build_generation_quality_bridge_matched_uncertainty_manifest.py`
- `scripts/run_generation_quality_bridge_terminal_uncertainty_waiter.py`
- focused builder and waiter tests

The builder rejects quality-result, source-report, sampling-protocol, real-set, checkpoint, sample-set, evaluator-runtime, or Git identity drift. The waiter uses the existing crash-safe `run_generation_stage_once.py` wrapper and the existing verified ImageNet-256 real feature cache.

## Verification

The targeted Windows suite covering quality-bridge construction, stage-once recovery, matched uncertainty, and both waiters passed. `compileall` and `git diff --check` also passed.

An isolated Linux checkout under `/tmp` replayed the same targeted suite with CUDA hidden. The first invocation used the conda `bin/python` symlink and was rejected by the intended no-symlink trust-boundary check; rerunning with the real `/root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10` executable passed. The isolated checkout remained tracked-clean and no GPU process was created.

Bundle identity:

- Path: `/tmp/cofitok-quality-bridge-terminal-uncertainty-e9a424b.bundle`
- Bytes: `48,306,828`
- SHA256: `acf4c8ff3d99560371ce93d4de26e2ea39b32801694318fe96ee94c12c7e2fb8`

The bundle was verified against the remote repository before deployment.

## Persistent deployment

Control checkout:

```text
/root/autodl-tmp/CoFiTok/checkouts/quality-bridge-terminal-uncertainty-e9a424b/CoFiTok-internal
```

Output root:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_terminal_matched_uncertainty_v1
```

The initial background launch, PID `844091`, omitted `PYTHONPATH` and exited at import time before acquiring the output lock or starting any GPU work. The only generated file was preserved as:

```text
reports/waiter.initial_pythonpath_failure.log
bytes: 321
sha256: a4cb6b819ccf4d933a1595943c92e62d020dce7d54035563cb2f4cac66574afc
```

The corrected launch explicitly bound the control checkout and its `src/` directory in `PYTHONPATH` and used the real `python3.10` binary. The active waiter is:

```text
PID: 844217
status: waiting
phase: quality_bridge
detail: waiting_for_quality_bridge_terminal_result
```

It is bound to:

- quality bridge revision `cf0e5faa94bf4ab38d947b921935b3b765b5537a`, tree `6cef27723196fd363379bca2e7b85b1678ebd777`;
- evaluator revision `1c8ef207cb6d79850d73a45abc345fc421e6aa7f`, tree `a09f14a0eca44af6db8e6781863a463163ddf646`;
- preceding frozen-pair uncertainty waiter PID `809569`, control revision `f161fe31453231b7c126d10cfbe78344f10ff463`;
- real feature cache bytes `409,601,577`, SHA256 `20103588dca9ce47bfceef6b68b473fdf4be720f149d1b8bdd96341d27c10dcd`.

Deployment receipt:

```text
reports/deployment_receipt.json
bytes: 3,260
sha256: cfe513e805b17f67067687100f846878cabcfb0b86b145e8aa7dcf9939ee256b
```

At the post-launch check, the active CoFiTok training had reached step 33,400. PID `619775` remained the only GPU compute process at 85,284 MiB. The new waiter had not created a terminal manifest, feature cache, or audit child and used no GPU memory.

## Scientific interpretation

This deployment does not itself prove a full-data generation advantage. It closes the statistical gap that would otherwise remain after the 100K pair finishes. A `pass` will support only a relative, matched-sample advantage for the exact bound checkpoints, sampling protocol, and sample-index window. A `hold` will state that the point-estimate direction was not confirmed under the paired uncertainty analysis. Absolute quality, recall, class fidelity, visual quality, and contextual comparison with official D-AR/MAR/ReTok rows remain separate questions.

Machine-readable evidence is stored at:

```text
artifacts/reports/generation/quality_bridge_terminal_uncertainty_waiter_deployment_2026-08-18.json
```
