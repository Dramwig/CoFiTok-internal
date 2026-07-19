# Generation deployment attestation

Date: 2026-07-20

Branch: `scale/generative-system`

## Problem

The initial controlled transition used a fixed receipt filename. Rank-layout
recovery then advanced the server through additional verified fast-forward
hops. Replacing the initial receipt would destroy history, while continuing to
read it as the terminal authority would bind the completion audit to an
obsolete target revision.

## Contract

`cofitok.generation_paths.generation_deployment_attestation_paths` derives all
deployment evidence paths from the full selected target SHA. The attestation
runbook refuses an abbreviated revision, a dirty or wrong branch, a target that
is not the deployed HEAD, an unrelated source revision, active generation
training, or an existing target receipt.

For the selected target it writes these persistent sources below
`checkpoints/generation/deployment/`:

- a prerequisite-aware aggregate Git bundle from the locked source revision;
- the source-to-target untracked-path conflict report;
- a full remote pytest JUnit report;
- a `git ls-files`-enumerated `bash -n` report for every tracked runbook;
- a schema-v2 receipt binding all source paths, byte counts, SHA256 values,
  revisions, Git state, and the locked training-pair validation.

The receipt is written last and cannot be overwritten by the runbook. The
terminal completion audit resolves the same target-specific paths from the
expected full-training revision and independently rehashes and revalidates all
four bound sources. Earlier deployment receipts and the preserved recovery
bundles remain unchanged.

The 2026-07-20 recovery bundles were copied byte-for-byte out of `/tmp` before
expiry. Their persistent files are:

```text
cofitok-generation-rank-recovery-05572b189548b14e963678f4174a989b408bc4d0.bundle
  bytes: 11520065
  sha256: b457808f78fcea5869bbae7742dfc43beaee3d6892fa3ee1506130d4900d07bf

cofitok-generation-probe-loss-hotfix-05bbb4af63a9f1d9b7f11bc4222d50875e382e1d.bundle
  bytes: 8496
  sha256: 8b1341da07c995aa1a20d42d561f84e5280967c28e7d15e1c1745857f392aee0
```

These hashes exactly match their original recovery receipts. They are ancestry
evidence; the future selected-target attestation is the terminal completion
authority.
