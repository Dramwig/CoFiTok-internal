# Capacity comparison statistical-claim addendum

Date: 2026-08-18 (Asia/Shanghai)

## Outcome

A separate, permanently non-authorizing claim-evidence addendum is now
implemented at revision `9dfa3c3e80d1f7d71056541eb8029d5627323162`.
It closes the remaining reporting gap between the capacity-full schema-v8
large-scale comparison and the terminal matched-bootstrap qualification.

The existing comparison schema and completion chain were intentionally left
unchanged. The addendum is post-terminal evidence: it cannot launch training,
sampling, export, release, or signal any process. This avoids a circular
dependency because the statistical qualification is produced only after the
capacity-full final gate already exists.

Machine-readable development evidence is stored at:

```text
artifacts/reports/generation/capacity_claim_evidence_addendum_2026-08-18/development_summary.json
```

## Bound claim

The addendum accepts only the `capacity_full` comparison profile. It requires:

- the exact capacity-full final gate and all comparison source identities;
- CoFiTok K=8 and `dense_identity` in the matched-training direct tier;
- ImageNet-256 at 256x256;
- 300,000 optimizer steps per method;
- 50,000 EMA samples per method;
- unchanged official-context source with cross-tier ranking disabled;
- a terminal statistical qualification bound to the same final gate;
- identical FID point estimates in the comparison, final gate, and uncertainty
  qualification;
- both the absolute quality gate and paired bootstrap uncertainty decision to
  pass before any positive claim is emitted.

When all checks pass, the only positive statement emitted is:

> Under the exact bound ImageNet-256 matched-training and matched-evaluation
> protocol, CoFiTok K=8 achieved a statistically supported lower FID than
> dense_identity.

The artifact always forbids broad generation superiority, SOTA, cross-tier
numeric ranking, and release or execution authority. A statistical `hold`
produces an explicit negative qualification instead of preserving the point
estimate as a claim.

## Rejection coverage

Focused tests cover rejection of:

- comparison or qualification SHA drift;
- a qualification bound to another final gate;
- FID drift between the comparison and uncertainty report;
- a forged authorizing claim boundary;
- cross-tier numeric ranking;
- quality evidence inconsistent with the physical final gate;
- malformed evaluator identity or changed embedded execution/uncertainty
  evidence;
- a positive qualification when CoFiTok does not have lower FID.

## Verification

Windows:

- focused comparison/qualification/waiter/post-eval suite: `44 passed`;
- broad capacity plus comparison suite: `244 passed`, with only the existing
  CRLF-sensitive frozen-SHA failure in
  `test_generation_capacity_recovery_supersession.py`.

Linux isolated rehearsal:

```text
/tmp/cofitok-capacity-claim-addendum-9dfa3c3-rehearsal
```

- exact revision: `9dfa3c3e80d1f7d71056541eb8029d5627323162`;
- focused suite: `44 passed`;
- broad capacity plus comparison suite: `245 passed`;
- Python compile: pass;
- `git diff --check`: pass;
- tracked state: clean;
- `CUDA_VISIBLE_DEVICES=-1` throughout test execution.

Incremental bundle:

```text
path:         /tmp/cofitok-capacity-claim-addendum-9dfa3c3.bundle
bytes:        12,587
sha256:       c97fff4ff70123f6eb6c11d39f7bb5b4cdb7ba921c546ed87d4123c95b3983f3
prerequisite: f7e3f9fdcdf7238eefaa3efac9a8215df6122e03
advertised:   9dfa3c3e80d1f7d71056541eb8029d5627323162
```

Remote bundle verification passed against the clean immutable receipt-builder
checkout. The formal remote checkout stayed at
`1ebcc15210e63a776a2ba448481cbd8bb94a4066` on `scale/generative-system`.

## Live boundary

At `2026-08-18T05:51:26+08:00`, the full-data quality bridge remained healthy:

- stage: `cofitok_training`;
- latest metrics step: `35,700/100,000` (`2,284,800` images seen);
- dense leg: not started;
- pair-monitor issues: none;
- only GPU compute PID: `619775`, about `85,286 MiB`;
- unrelated GPU compute: none;
- free `/root/autodl-tmp` capacity: `324,348,026,880` bytes.

This implementation does not itself prove a generation advantage. It ensures
that if the capacity-full experiment later passes its final quality gate and
paired uncertainty audit, the resulting relative FID claim is exact,
source-bound, narrow, and auditable.
