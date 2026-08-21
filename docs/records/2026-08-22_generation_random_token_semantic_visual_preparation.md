# Exact 100K random-token semantic visual diagnostic preparation

Date: 2026-08-22

## Purpose

The full-data 100K checkpoint evaluator already records random-token component
energy, exact zero-token synthesis, and shuffled-token mismatch. Random-token
energy alone does not establish that the restricted synthesis operator avoids
obvious natural-image semantic structure. This change prepares a small,
source-bound visual diagnostic for the exact CoFiTok 100K EMA checkpoint.

The diagnostic is intentionally narrower than a generation-quality evaluation.
It applies `S_k` directly to independent standard-normal token fields, renders
every dense component and cumulative prefix with a symmetric RMS scale, and
places them beside independent pixel-Gaussian controls matched to each
component's per-sample RMS. It does not use `x_t`, timestep, class labels,
previous tokens, U-Net features, or the token predictor when producing the
random-token panels.

## Exact checkpoint target

- training revision: `cf0e5faa94bf4ab38d947b921935b3b765b5537a`
- training tree: `6cef27723196fd363379bca2e7b85b1678ebd777`
- training branch: `scale/generation-stability-quality-bridge-100k`
- checkpoint step: `100000`
- checkpoint bytes: `1010937514`
- checkpoint SHA256:
  `b36a92229ba2dd021db9c7585970ddda1d17d1919ca6b0eec0362b06d4bd462e`
- integrity sidecar bytes: `597`
- integrity sidecar SHA256:
  `d5dd3ac3afb9c5e1d6facf3ea766700c221f17b22724749056213a55dc19c2d5`
- physical integrity audit SHA256:
  `c364a157d3ee1c9df798f0b25fd34da8ab3cc6115fffca93242d925554b442c2`

The evaluator uses the shared `load_generation_model` trust boundary, so the
physical checkpoint is hashed and verified against the adjacent integrity
sidecar before deserialization. The CLI additionally requires the exact
checkpoint SHA256 and step.

Before checkpoint deserialization, the Python entrypoint itself now validates
the completed terminal quality result, terminal system guard, factorization
supervisor status, exact requested checkpoint path/SHA/step, and an empty
`nvidia-smi` compute-process set. These checks no longer rely only on the shell
runbook. The three terminal JSON sources and evaluator Git/source identities are
rehash-checked again after checkpoint load and before the final report is
published, closing the relevant runbook-to-entrypoint and evaluation-time
TOCTOU windows.

## Evidence and claim boundary

The report records:

- evaluator revision/tree/branch and evaluator-source SHA256;
- checkpoint bytes/SHA256/step and integrity-sidecar identity;
- token layout and aggregate compression;
- synthesis module/parameter/buffer inventory;
- trainable parameter, bias, nonlinearity, and attention checks;
- exact zero-token and cross-token-isolation checks;
- a numerical linearity check;
- random-token and matched pixel-Gaussian component/prefix statistics;
- content-addressed PNG panel identities.

The report permanently states:

```text
non_authorizing=true
full_training_launch_allowed=false
full_300k_launch_allowed=false
release_authorization_allowed=false
promotion_gate_substitute=false
semantic_absence_proven=false
generation_quality_advantage_proven=false
```

The visual result can screen for obvious learned semantic rendering by `S_k`,
but it cannot prove that every possible random token lacks semantic structure.
It also cannot establish generation-quality superiority over `dense_identity`.

## Execution boundary

The runbook refuses to execute until:

1. the terminal `quality_bridge_result.json` exists with `status=completed`;
2. its non-authorizing boundary remains exact;
3. the evaluator checkout matches the explicitly supplied revision/tree/branch
   and has no tracked changes;
4. the exact 100K checkpoint and sidecar exist;
5. `nvidia-smi` reports no active compute PID;
6. the terminal system claim guard is complete (`pass` or `hold`);
7. the conditional factorization-quality supervisor is terminal and has no
   child process;
8. the dedicated runbook `flock` is acquired;
9. an existing output root, if any, passes exact manifest-bound `--resume`
   validation.

Output is isolated at:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_random_token_semantic_visual_v1
```

The current dense 100K training and terminal evaluation chain must finish first.
This preparation does not launch the diagnostic and does not modify the formal
remote checkout.

## Local verification

- focused tests: `42 passed`
- full sibling-layout suite: `1087 collected / 1081 passed / 6 skipped`
- Python compile: passed
- `git diff --check`: passed
- pro6000 native Linux runbook `bash -n`: passed

The focused set covers the new diagnostic, production checkpoint evaluation,
legacy synthesis diagnostics, and synthesis operators.

The earlier flat `D:/cofitok-random-token-visual` full-suite attempt had four
expected paper-layout failures because legacy tests resolve the outer sibling as
`D:/paper`. No implementation test failed. A temporary, verified junction to the
canonical project `paper/` directory was used for the final sibling-layout
replay after the Python-entrypoint hardening; all `1081` runnable tests reached
terminal success, `6` were skipped, and the junction was removed while the
canonical paper directory remained untouched.
