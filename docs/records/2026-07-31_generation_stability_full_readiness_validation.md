# Stability-full readiness split validation (2026-07-31)

## Implementation identity

- Branch: `scale/generation-large-capacity`
- Readiness/launch split commit:
  `50145a9bf733b0fb7edf4a342883c43ea3fde307`
- Formal server repository remained:
  `1ebcc15210e63a776a2ba448481cbd8bb94a4066` on
  `scale/generative-system` with a clean tracked worktree.

## Local validation

The project-local `.venv` full suite completed with no failures:

- collected node IDs: `822`;
- progress output: all tests passed except the existing three skips;
- `git diff --check`: pass before commit.

The focused readiness, runbook, entrypoint, and completion-audit suite completed
`20/20` before the final full run. Coverage includes:

- exact 250M config and parameter contracts;
- schema-v2 storage reserve and 4x arithmetic;
- fixed five-candidate runtime grid and `1x64` baseline;
- source-byte drift and deterministic readiness replay;
- refusal when formal training state already exists;
- full launch containing no runtime selector;
- completion audit binding the externally supplied readiness SHA256.

## Rehearsal bundle

The isolated rehearsal used the implementation-commit bundle:

`/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment_bundles/
cofitok-generation-large-capacity-50145a9-from-1ebcc15.bundle`

- bytes: `39,353,953`;
- SHA256:
  `fb6f1497f69a79eb143c04954e27eaad808de1d2292a1509c9e688ebcf8a6a34`;
- advertised head:
  `50145a9bf733b0fb7edf4a342883c43ea3fde307`;
- prerequisites:
  `1ebcc15210e63a776a2ba448481cbd8bb94a4066` and
  `58d83bfce2770eab2565b8c89a5f9a06201a0c86`.

The formal server repository itself successfully ran `git bundle verify`.
An earlier 36,849-byte bundle based only on `c1efb12` was rejected by the
formal repository because that prerequisite object is absent there; it was
removed from the persistent deployment-bundle directory to prevent misuse.
After this validation record was committed, the implementation bundle was
superseded by a new bundle advertising the branch's final handoff HEAD. The
authoritative current bundle path, bytes, SHA256, advertised head, and
prerequisites are intentionally maintained in the repository-external project
`AGENTS.md` handoff so recording them does not recursively change the Git head
they identify. Do not use the rehearsal bundle as the current deployment input.

## Isolated Linux rehearsal

Rehearsal root:

`/tmp/cofitok-generation-large-capacity-readiness-50145a9/`

The checkout was cloned from the formal repository, fetched only from the
verified bundle, and detached at exact commit `50145a9`. The sibling paper tree
was linked at the expected parent layout. CUDA was disabled for tests.

- JUnit: `814 tests, 0 failures, 0 errors, 2 skipped`;
- JUnit SHA256:
  `e73de9b9b13198f24fc3138ee8f4e48c88a715918ddea1bf59ab4e3570151abd`;
- all `96/96` tracked shell runbooks passed `bash -n`;
- rehearsal tracked worktree: clean;
- formal repository HEAD/branch/tracked state: unchanged and clean.

## Active training boundary

No CUDA readiness benchmark or full 300K training was started. During
validation, the only GPU process was active trainer PID `319202` using about
`84,122 MiB`. The scaling monitor remained
`running / cofitok_training / issues=[]`; CoFiTok had reached step
`10,050/50,000` and dense identity remained at step `0/50,000`.

The new readiness runbook must therefore remain dormant until the source-bound
50K promotion gate passes and the GPU is explicitly available. This validation
proves implementation and deployment readiness only; it does not authorize or
claim full 300K generation completion.
