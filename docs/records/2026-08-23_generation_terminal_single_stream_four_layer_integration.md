# 2026-08-23 terminal single-stream four-layer integration

## Scope

This isolated evidence branch integrates the terminal single-stream replication
boundary across four CPU-only consumers:

1. terminal strong-baseline comparison;
2. terminal completion audit;
3. statistical claim-language guard;
4. terminal system claim guard.

The scientific boundary remains exact:

- `bound_terminal_stream_count=1`;
- `independent_replication_count=0`;
- `independent_replication_supported=false`;
- FID direction, paired block-KID, bootstrap, and sign-test results are analyses
  of one bound terminal 10K stream, not independent replications.

## Git lineage

- repository: `D:/cofitok-terminal-single-stream-integration`;
- branch: `analysis/generation-terminal-single-stream-integration-v1`;
- completion boundary base: `8dc4884910aa068978b54af4c0a055813d89e895`;
- statistical guard integration: `6512ef9fe0287e1233d0a479b0e01d48be677345`;
- terminal guard integration: `8e86fc3c37d29fe0420bb083c757b3ad6fb8575e`;
- dependency-closure implementation:
  `15698e77298a2907fefc08c3317987ea14b3fc27`;
- implementation tree: `b0c8e6bbc6c6b35f77044d5aeb59914160052178`.

## Dependency closure

The integration base intentionally lacked the quality-bridge dependencies that
the statistical and terminal guards import. Seven scripts were restored from
the clean classifier-integrity source checkout
`analysis/generation-terminal-classifier-integrity-v1-20260822@4087f4293e3fd197c81cbfa9029f3d6083654415`:

- `audit_generation_matched_uncertainty.py`;
- `build_generation_matched_uncertainty_summary.py`;
- `build_generation_quality_bridge_matched_uncertainty_manifest.py`;
- `build_generation_quality_bridge_statistical_claim_qualification.py`;
- `build_generation_requested_class_visual_audit.py`;
- `calibrate_generation_class_fidelity_classifier.py`;
- `run_generation_matched_uncertainty_waiter.py`.

All seven committed Git blob IDs are byte-identical to that source checkout.

The statistical guard no longer eagerly imports the 300K capacity addendum.
`build_generation_capacity_claim_evidence_addendum` is loaded only inside the
`capacity_full_300k` source path. Therefore the authorized 100K quality-bridge
consumer path does not depend on a forbidden 300K module. If the optional
capacity path is requested without its dependency, the guard fails closed with
an explicit runtime error. Tests cover both an injected valid capacity contract
and the absent-dependency failure.

## Verification

All checks used the project-local Python environment, with CUDA hidden and
`OMP_NUM_THREADS=MKL_NUM_THREADS=1`.

- complete pytest collection: `1195 tests collected`;
- four-layer and adjacent clean-Git/checkpoint-resume regression:
  `88 passed`;
- complete code suite excluding only the four outer-paper-layout tests:
  `1184 passed, 7 skipped in 383.02s`;
- the four excluded tests require the outer sibling paths
  `D:/paper/venues/aaai27/main.tex` and `D:/paper/latex/main.tex`, which are not
  part of this isolated inner-repository checkout; their test source is
  normalized byte-identical to the main workspace copy, and the authoritative
  paper files exist under
  `C:/Users/17194/Desktop/PaperFiled/CoFiTok/paper/`;
- all integrated Python entrypoints compiled successfully;
- `git diff --check` passed;
- the implementation checkout was tracked clean for the clean-Git waiter
  regression.

## Live-chain boundary at verification time

The remote chain was re-read rather than inferred from prior summaries:

- CoFiTok: exact `100000/100000`, `6400000` samples;
- dense: `96700/100000`, `6188800` samples;
- both metrics files strictly increasing with `samples_seen=step*64`;
- pair monitor: `running`, `issues=[]`;
- sole GPU owner: dense trainer PID `219593`;
- controller PID `219486` held both the recovery lock and canonical
  `quality_bridge_execution.lock` through file descriptors 3 and 4;
- dense 100K physical audit and replay were still waiting;
- terminal comparison/completion outputs did not yet exist;
- `generation_advantage_proven=false`.

## Authorization boundary

This integration is local, CPU-only, and permanently non-authorizing. It was
not deployed and did not replace or signal any live waiter. It does not permit
training, sampling, a second controller or trainer, full 300K launch,
promotion, export, release, or process signals. The live canonical chain and
its standing authorization were unchanged.
