# Capacity generation control-plane continuity archive

Date: 2026-08-15

## Outcome

The active quality-bridge-to-capacity generation chain now has a persistent,
source-bound cold-recovery archive for the static assets that previously existed
only under `/tmp`. The final v2 archive is self-contained relative to its four
declared Git bundles, and a full materialization rehearsal reconstructed all
three vulnerable checkouts and five non-Git runtime files without launching or
signaling a process.

This is continuity evidence only. It does not authorize training, sampling,
evaluation, promotion, export, or release.

## Recovery gap found by rehearsal

The first v1 archive contained the complete-history follow-up bundle, the
quality-bridge execution bundle, and the capacity-preparation bundle. Its
byte-level verification passed, but the first cold-restore rehearsal correctly
failed when Git reported that the capacity-preparation bundle required the
missing commit:

```text
c7424ed74da1bd7907d094d10725b337b1d822ac
```

The incomplete evidence was retained rather than overwritten:

- v1 manifest: `8,427` bytes, SHA256
  `b5869a655626dc235f238b40aad5fcc41b508b587e426ce80ea1d3e4735c2184`;
- machine-readable closure failure: `1,651` bytes, SHA256
  `72c3e7b0132a0afc1e7496d6431fa2e9255324e6845d1ffbb27b2231bcd61803`;
- failed rehearsal tree:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/control_plane_continuity/capacity_generation_pipeline_v1_restore_rehearsal`.

The final implementation adds the exact capacity-reference bundle and validates
the whole declared bundle sequence in a temporary bare repository before the
archive directory is atomically published. An incomplete closure now removes
its bounded staging directory and cannot become an accepted archive.

## Exact implementation identity

- branch: `scale/generation-capacity-control-continuity-v1`;
- final revision: `a89c1fa29fda494f3492d32dfa5e9a60bfa705a9`;
- final tree: `e460f1d67ba86a2a5c21998c84729fe12dd682a2`;
- isolated checkout:
  `/root/autodl-tmp/CoFiTok/checkouts/capacity-control-continuity-a89c1fa/CoFiTok-internal`;
- v2 plan: `6,288` bytes, SHA256
  `c523e27db322daaf6db392f6645915ad076226f65858707743614623884f54be`.

The compact persistent deployment bundle directly spans the already-deployed
lineage-observer revision to the final continuity revision:

- path:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/control_plane_continuity/capacity_generation_pipeline_v2/deployment/cofitok-capacity-control-continuity-a89c1fa-from-0bbd1cc.bundle`;
- bytes: `19,789`;
- SHA256:
  `1fc585377e3b83accf865ed3e0b6f66382237073aa0f2ae73ff08ffbb1276e1d`;
- prerequisite: `0bbd1cc43b9bb713ab2da8d4da6404f304f88ffe`;
- advertised head: `a89c1fa29fda494f3492d32dfa5e9a60bfa705a9`.

## Final archived payload

The v2 archive root is:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/control_plane_continuity/capacity_generation_pipeline_v2
```

Its manifest is `9,413` bytes with SHA256
`359744d431d758ad503bb2404e97091533477cb353a2bbbef5a09904b52b55b8`.
The nine payload files are:

| role | bytes | SHA256 |
|---|---:|---|
| follow-up complete-history bundle | 137,665,447 | `67839784ac4a2c2ef9e0c5723d55703bae6ba20ffc12fb03e214bf7512953861` |
| quality-bridge execution bundle | 40,130,783 | `3392e2a66c48237b8bfa0bde519245d4d17d055bb21b4e86321537fe5c3a871a` |
| capacity-reference prerequisite bundle | 14,553 | `689c82b4c5a729cce8e567f3bdf365577b9e8a5e02023236c8c40869b828f447` |
| capacity-preparation bundle | 19,459 | `e8341f11288d37c95e1934afa2a5d899ce9d96929c9fe070945693eae1f5a03b` |
| standing authorization | 865 | `5fe64a0941acb55a21cb6479a726987c6259e93a772c47290f2d6883752de4df` |
| execution approval | 1,754 | `e9da52a4e7ff1b4700b70aadaa8703ee40fb1a9862e847dcfb6e75933d295a4b` |
| bounded recovery supervisor | 33,131 | `008a9d83e59fa433d2d4542dbb3b3e29ec8e5bee8eadf5190f5b3b1c99b40cbf` |
| idle waiter | 11,268 | `66dca29d14fd1cf4f7bd8129622a6734e12ab3c06a4b1a70d7bd45fc77e80a4e` |
| follow-up waiter | 10,743 | `77859b4bdcff20e79449654a1b20a42d6ad21d1c96bec35e189b01b22f01edd2` |

The verified Git fetch order is:

1. `9b02fa83d20b1459a2706d6d82371caf5c023f54` complete history;
2. `cf0e5faa94bf4ab38d947b921935b3b765b5537a` quality-bridge execution;
3. `c7424ed74da1bd7907d094d10725b337b1d822ac` capacity reference;
4. `22a5994f0475c29cca38a80188ed6ce00c6e6d15` capacity preparation.

The final independent verification report is `2,931` bytes with SHA256
`4481c1c184ca636dfa9ff35379c7bc180518d5874b714361a55f1fba6a5587bd`.

## Cold-restore proof

The successful rehearsal target is:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/control_plane_continuity/capacity_generation_pipeline_v2_restore_rehearsal
```

The `9,747`-byte restore report has SHA256
`478b1fb74a7078534fa4773e06cde89e0c7a36f9587d6be554a5fa89e720e5a6`
and proves:

- all four bundle bytes and advertised revisions were restored;
- all three checkout revisions, trees, branches, and clean tracked states match;
- all five runtime-file bytes, SHA256 values, and modes match;
- no experiment or background process was launched;
- no process was signaled;
- no GPU was queried or allocated;
- the formal checkout was not modified;
- no execution authorization was created.

No bundle-closure temporary directory remained after validation.

## Validation and live-state preservation

- exact final revision continuity, lineage, and full runbook-entrypoint suite:
  `17 passed` locally;
- exact final revision targeted continuity suite: `7 passed` on Linux with
  CUDA hidden;
- Python compilation, runbook `bash -n`, `git diff --check`, and clean tracked
  Linux checkout: passed.

At the final live recheck all 16 pre-existing blocking/auxiliary processes plus
the lineage observer remained alive. The observer had no issues and continued
to identify `quality_bridge_100k_recovery / waiting_for_gpu_idle` as the first
blocker. The only GPU process remained unrelated FieldScope PID `910099` at
about `2,256 MiB`; it was not signaled or shared.

The formal checkout remained exactly:

- revision: `1ebcc15210e63a776a2ba448481cbd8bb94a4066`;
- branch: `scale/generative-system`;
- porcelain count: `87`;
- porcelain SHA256:
  `a7e1a2daf19f77e5d92efaa09b2768f36f5df2e989475480f7d22e54b46e9497`.

No capacity-full training launch receipt or 300K training output was created by
this continuity work.
