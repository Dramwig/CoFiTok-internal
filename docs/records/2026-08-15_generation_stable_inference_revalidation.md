# Stable generation inference revalidation

Date: 2026-08-15

## Outcome

The stable inference, resumable sampling, EMA deployment-artifact, and terminal
completion-receipt contracts were revalidated on both Windows and Linux while
the production GPU remained unavailable. No source change was required.

This verifies the executable failure boundaries of the inference system. It is
not evidence that a production 300K artifact exists, does not replace formal
50K generation metrics or the final quality gate, and creates no training,
sampling, promotion, export, or release authorization.

## Covered behavior

The 61-test bounded suite covers:

- resumable EMA-only export with immutable manifests, output locks, atomic
  publication, and refusal to repair unbound or drifted partial state;
- artifact, sidecar, source-training authorization, final-release gate, Git,
  and runtime-environment provenance checks before model deserialization;
- terminal completion receipts, including terminal-audit drift rejection and
  valid artifact consumption after the large source training checkpoint and
  its integrity sidecar have been archived;
- repeated `GenerationSession` requests, immutable request validation,
  deterministic per-seed streams, metadata propagation, finite-output checks,
  and the inference CLI's atomic report/manifest/progress chain;
- exact sampling and inference resume, read-only completed replay, regeneration
  of only missing or digest-mismatched PNG identities, unexpected-output
  refusal, and interrupted PNG encoding without a published partial file;
- checkpoint-to-session sampling preflight, release/completion authorization,
  checkpoint integrity failure, and CPU execution without CUDA availability.

The consumer boundary was inspected directly: completion authorization rehashes
the terminal audit, recreates the receipt expectation map, rehashes the bound
export manifest and physical artifact, and checks its integrity metadata before
`torch.load`. Routine consumption does not reopen the large source training
checkpoint, but it still requires the immutable audit, receipt, export manifest,
artifact sidecar, and EMA artifact bytes.

## Exact Linux evidence

- checkout:
  `/root/autodl-tmp/CoFiTok/checkouts/capacity-control-process-rehearsal-47e59a0/CoFiTok-internal`;
- revision: `47e59a0ce926c3ac1fdaa4860de84d2f27e39f76`;
- tree: `3a99993b66db675f262ed2e04298e1d4d64411cf`;
- branch: `scale/generation-capacity-control-process-relaunch-rehearsal-v1`;
- tracked checkout state: clean;
- Python: `3.10.20`;
- CUDA policy: `CUDA_VISIBLE_DEVICES=''`;
- result: `61 passed`, zero failures/errors/skips;
- persistent JUnit time: `118.003` seconds.

The persistent JUnit report is:

- path:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/control_plane_continuity/stable_inference_revalidation_2026-08-15/pytest_linux_cuda_hidden_47e59a0.xml`;
- bytes: `10,319`;
- SHA256:
  `00826a681193b08465ac36d29c90c865e48a12258caaada7306c66aec51a48f7`;
- file mode: `0644`.

An initial Linux run of the same suite also passed `61/61` before the persistent
report run. The local Windows project environment passed `61/61` in 42.1
seconds. The four exact test modules were:

```text
tests/test_generation_inference_artifact.py
tests/test_generation_session.py
tests/test_generation_sampling.py
tests/test_generation_sampling_preflight.py
```

## Preserved production state

The validation used temporary CPU-only fixtures and no production checkpoint.
At the final pre-record check, the only GPU process remained unrelated
FieldScope PID `910099` at approximately `2,256 MiB`; the quality-bridge
recovery supervisor remained at attempt `0`, waiting for five consecutive idle
polls. The formal CoFiTok checkout was not modified.
