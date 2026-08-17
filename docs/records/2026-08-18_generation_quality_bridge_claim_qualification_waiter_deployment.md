# Full-data 100K statistical-claim qualification waiter deployment

Date: 2026-08-18 (Asia/Shanghai)

## Outcome

A permanently non-authorizing waiter now closes the final reporting gap for the
active full-data ImageNet-256 100K matched quality bridge. It waits for the exact
terminal quality result and the source-bound paired uncertainty report, then
builds one statistical-claim qualification artifact.

The waiter can emit a positive statement only when both inputs pass:

> Under the exact bound full-data ImageNet-256 100K matched protocol, CoFiTok
> K=8 achieved a statistically supported lower FID than dense_identity.

Any quality or uncertainty failure produces `hold`. The waiter cannot launch
training or sampling, authorize 300K or release, export a model, signal a
process, replace either source result, or support a broad generation-superiority
or SOTA claim.

Machine-readable evidence is stored at:

```text
artifacts/reports/generation/quality_bridge_claim_qualification_waiter_deployment_2026-08-18.json
```

The identical authoritative remote receipt is:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/
stability_full_data_100k_terminal_claim_qualification_v1/
reports/deployment_receipt.v2.json

bytes:  14,699
mode:   0444
sha256: ce103e7952da4bbd52a861dc5d3ea0716e2d00c393b80097b973d2f55b7ff457
```

The first published receipt remains immutable at
`reports/deployment_receipt.json` (`14,029` bytes, SHA256
`007d70c7ec0b5c811f4db1cf2d395b69cf99b1354c7152f815dd25dede2af59f`).
It used one overbroad combined boolean for the failed launch. The waiter had
acquired and released its own exclusive output lock before contract validation
failed, while it launched neither a terminal child nor GPU work. Revision 2
preserves the first receipt identity in `supersedes` and narrows only that field;
all code, bundle, process, source, output, and claim-boundary identities remain
unchanged.

## Locked implementation

```text
branch:   analysis/generation-quality-bridge-claim-qualification-v1
revision: c42ac96c6628ff71f7c67c8957c86ea0523aea0b
tree:     f94f2deb974b0a347b3a647e68dc3dcba22a0d63
subject:  Bind quality tree in claim waiter
```

Persistent control checkout:

```text
/root/autodl-tmp/CoFiTok/checkouts/
quality-bridge-claim-qualification-013beb7/CoFiTok-internal
```

The checkout is tracked-clean. The formal remote checkout was not modified and
remained tracked-clean at revision `1ebcc15210e63a776a2ba448481cbd8bb94a4066`
on `scale/generative-system`.

Source identities:

```text
build_generation_quality_bridge_statistical_claim_qualification.py
bytes:  15,602
sha256: cbed88c3ca19cdadbd233a0c3073b293226107d0979151c851ae7d48eca5ea08

run_generation_quality_bridge_claim_qualification_waiter.py
bytes:  19,109
sha256: 126a1aeafc592ac24b7a44f42995d24fe4dfc099e21fb044e09234d5cf435d6f
```

## Bundle chain

Initial bundle:

```text
path:         /tmp/cofitok-quality-claim-waiter-deploy-013beb7.bundle
bytes:        17,041
sha256:       353b0964be48a00dfae9b54e4d4972804a7cab2e4ccffd1c13144cadd164311c
prerequisite: 9dfa3c3e80d1f7d71056541eb8029d5627323162
advertised:   013beb7cafde7805a1b06825d2318c5d0bdab71e
```

Quality-tree hotfix bundle:

```text
path:         /tmp/cofitok-quality-claim-waiter-hotfix-c42ac96.bundle
bytes:        2,900
sha256:       7170d7c40149623f708c384aff5c030b91af3c9aeabe817ede5e4ac05954c98e
prerequisite: 013beb7cafde7805a1b06825d2318c5d0bdab71e
advertised:   c42ac96c6628ff71f7c67c8957c86ea0523aea0b
```

Both bundles passed `git bundle verify` in the deployed control checkout, and
the hotfix was applied by fast-forward.

## Preserved initial failure

The initial process, PID `877549`, exited during source-contract validation. It
did not use the GPU, acquire a terminal child, launch sampling, or affect the
trainer or upstream waiter.

The upstream terminal waiter includes `quality_git.tree`. Initial revision
`013beb7` bound only the quality revision and branch, so it correctly failed
closed with:

```text
ValueError: quality-bridge terminal waiter contract differs
```

The original evidence remains preserved:

```text
waiter.initial_quality_tree_contract_failure.json
bytes:  1,093
sha256: bd063b0b73e8a46f8453cc34af83eb628e5c8f7da1f31c7e8671ff3ae2b173b1

waiter.initial_quality_tree_contract_failure.log
bytes:  1,292
sha256: c8c106e70c81846c17e358bf2e941db742a6df30907e30c255f4d96fe925a02e
```

Revision `c42ac96` added the missing exact tree binding and the
`--expected-quality-tree` launch argument. The failed PID was not reused and no
training or upstream waiter was restarted.

## Final validation

The final `c42ac96` checkout replayed the full related Linux suite with
`CUDA_VISIBLE_DEVICES=-1`, `OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`:

```text
tests/test_generation_quality_bridge*.py
tests/test_generation_matched_uncertainty.py
tests/test_generation_capacity_statistical_claim_qualification.py
tests/test_generation_capacity_terminal_uncertainty_waiter.py

83 collected
83 passed
0 failed
0 skipped
```

The final revision and tree matched the deployed waiter after the suite, and
tracked status remained empty. The Linux GPU compute table continued to contain
only the active trainer PID `619775`; no validation process used the GPU.

Earlier validation retained for provenance:

- Windows broad related suite: `82 passed + 1 skipped` (`83 collected`).
- Linux broad suite at initial revision `013beb7`: `83 passed`.
- Final hotfix targeted Windows/Linux suite: `17 passed`.

## Active deployment

At receipt time the process chain was:

```text
PID 878415  bash nohup wrapper, PPID 1
PID 878416  claim-qualification waiter, PPID 878415
```

The wrapper was retained without signaling. The actual waiter used the real
Python 3.10 executable and was bound to:

```text
status: waiting
phase:  source
detail: waiting_for_terminal_quality_and_uncertainty_sources

CUDA_VISIBLE_DEVICES=-1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

Its exact argv, environment, PID/PPID, checkout, source paths, output paths,
Git identities, and status/pid-file hashes are embedded in the machine-readable
receipt.

The upstream terminal uncertainty waiter remained:

```text
PID:    844217
status: waiting
phase:  quality_bridge
detail: waiting_for_quality_bridge_terminal_result
```

At receipt time none of the four terminal source artifacts existed yet:

- `quality_bridge_result.json`;
- `terminal_100k_execution_manifest.json`;
- `terminal_100k/matched_uncertainty.json`;
- `reports/statistical_claim_qualification.json`.

This is the expected pre-terminal state.

## Live training boundary

At `2026-08-18T06:31:00+08:00`:

- pair monitor: `running / cofitok_training`, `issues=[]`;
- CoFiTok: step `36,550/100,000`, `2,339,200` images seen;
- dense identity: not started;
- only GPU compute PID: `619775`, about `85,286 MiB`;
- unrelated GPU compute: none;
- free `/root/autodl-tmp` space: `323,921,149,952` bytes.

This deployment strengthens provenance only. It does not change the current
scientific conclusion: the ordered prefix-denoising mechanism is supported,
while the full-data relative generation claim remains unproven until the 100K
quality result and paired uncertainty audit both terminate successfully.
