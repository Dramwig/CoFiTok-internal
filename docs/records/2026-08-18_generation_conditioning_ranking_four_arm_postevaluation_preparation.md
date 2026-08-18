# 2026-08-18 conditioning-ranking four-arm held-out postevaluation preparation

## Purpose

The prepared training-time class-ranking probe cannot be interpreted from its
hinge-loss rows alone. This follow-up adds the missing CPU-only, held-out,
four-arm postevaluation that distinguishes a real semantic-alignment recovery
from a training-metric change.

The four source arms remain:

- control CoFiTok K8;
- ranked CoFiTok K8;
- control dense identity;
- ranked dense identity.

The immutable training candidate is:

- revision: `7d5ff8c661ee3f17c99b2c5fb17525051b27a92f`;
- branch: `scale/generation-label-ranking-probe-v1`;
- output root:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/conditioning_ranking_four_arm_probe1k_v1`;
- Linux preparation report:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/preparations/conditioning_ranking_four_arm_probe1k_v1/7d5ff8c661ee3f17c99b2c5fb17525051b27a92f/preparation.json`;
- preparation report bytes: `8,686`;
- preparation report SHA256:
  `cbdb9e7bda3a7b132e5e5b32187e6a6cfc2f163ec412b2a32769d8907d8986a6`.

No four-arm training output exists at preparation time. This record and the
postevaluation implementation do not launch that training.

## Held-out evaluation contract

The postevaluation uses the first validation image for ImageNet labels 64--71,
which are outside the labels 0--7 used by the earlier diagnostic. Every arm is
evaluated with the same:

- EMA weights;
- eight images;
- timesteps `100`, `500`, and `900`;
- wrong-label offset `500`;
- base noise seed `204060`;
- two CPU threads.

Only `t=500` and `t=900` enter the decision because the ranked training loss is
active only at timesteps greater than or equal to `500`. The `t=100` rows are
retained as descriptive evidence. Timesteps are averaged within each held-out
image before any sign test; the independent unit is therefore the image, not a
timestep row.

For each method, a recovery requires all of the following:

1. the ranked arm has positive mean correct-vs-wrong and correct-vs-null raw-MSE
   advantages;
2. both absolute ranked advantages pass a one-sided image-level sign test with
   `p < 0.05`;
3. the ranked-minus-control improvement is positive for both references;
4. both ranked-minus-control comparisons pass the same image-level sign test;
5. ranked correct-label MSE divided by control correct-label MSE is at most
   `1.02`.

Shared recovery requires both CoFiTok and dense identity to pass. A one-method
pass is treated as asymmetry, not as a shared repair and not as a CoFiTok
advantage claim.

## Fail-closed provenance checks

`scripts/build_generation_conditioning_ranking_probe_posteval.py` validates:

- the exact immutable preparation revision, branch, output root, config hashes,
  ranking contract, and parameter counts;
- four exact 1K training reports and their complete schema-v2 audits;
- checkpoint filename, bytes, SHA256, integrity sidecar, dataset identity,
  runtime identity, and clean training Git provenance;
- exact resolved configs, validation-event count, checkpoint retention, and all
  rollout/EMA-teacher/ranking schedule audits;
- a clean held-out evaluator checkout and CPU-only sensitivity reports;
- exact dataset, label-map, image identities, model identities, requests, and
  all 24 row identities per arm;
- finite positive raw correct/wrong/null epsilon MSE values;
- agreement of every derived ranking field with a fresh recomputation from raw
  MSE values;
- identical held-out rows across all four arms;
- exact output reuse on `--resume`.

The runbook is:

`artifacts/runbooks/generation_conditioning_ranking_four_arm_posteval_v1.sh`

It hides CUDA, limits CPU threads, uses `nice` and idle I/O priority, refuses to
run while a four-arm trainer is active, rechecks preparation/training-status
hashes after all sensitivity evaluations, and makes completed JSON evidence
read-only.

## Claim and authorization boundary

This stage is permanently diagnostic-only. It:

- generates no diffusion samples;
- changes no checkpoint;
- authorizes no training, sampling, checkpoint promotion, full training, or
  release;
- does not replace the formal image-quality or class-fidelity gates;
- never permits a CoFiTok-specific advantage claim from a shared-repair result.

Even a shared pass only supports considering a separately authorized matched
generated-sample validation.

## Validation state

At preparation time:

- `git diff --check`: pass;
- Python compileall for the new builder/tests: pass;
- targeted conditioning ranking/sensitivity/postevaluation suite: `26 passed`.
- complete Windows code suite from the D-drive isolated worktree, excluding only
  the repository-topology-dependent paper-layout file: `1,113 passed, 6 skipped`;
- the excluded paper-layout file from the canonical project root: `4 passed`.

The split invocation is necessary because that legacy paper test resolves
`paper/` relative to the worktree's parent, while the D-drive worktree is
intentionally outside the local paper workspace. Together the two invocations
cover the complete Windows suite.

The isolated `pro6000` Linux rehearsal also completed with CUDA hidden, two CPU
threads, `nice -n 19`, and idle I/O priority:

- incremental bundle prerequisite verification: pass;
- clean exact checkout and tree verification: pass;
- Python compileall: pass;
- runbook `bash -n`: pass;
- targeted conditioning suite: `26 passed`;
- complete Linux code suite excluding the same local-paper topology test:
  `1,117 passed, 2 skipped`;
- formal remote checkout: not moved or modified.

The paper topology assertions were already covered by the canonical-root
Windows invocation above. The evaluator implementation is therefore ready for
CPU execution after the separately controlled four-arm training stage produces
its four completed reports and audits.
