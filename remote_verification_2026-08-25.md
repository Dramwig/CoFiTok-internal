# Min-SNR Terminal Guard Remote Verification

Date: 2026-08-25 19:29 CST

This is an isolated, CPU-only verification record for the committed
non-authorizing terminal physical replay guard. It is deliberately outside
the active training checkout and does not authorize or launch a terminal
evaluation.

## Guard identity

- source branch: `analysis/generation-min-snr-terminal-physical-replay-v1-20260825`
- source revision: `05f18ae8855f756a422f79283d304268efaea03d`
- source tree: `7baded178b62d354010090198381522518723a82`
- incremental bundle: `D:/cofitok-bundles/min-snr-terminal-physical-replay-05f18ae.bundle`
- bundle bytes: `23994`
- bundle SHA256: `aa4a8e9d70bb6339dc6e00662a8844aec379b9ec1b5934f3324a98db3715ef85`
- prerequisite: `842a34130e82f241330707118a05bf6ed01e263e`

The bundle verified locally and on `pro6000`. The remote isolated checkout
was cloned from the authoritative training checkout, fetched only the bundle,
and checked out the advertised guard branch. The authoritative training
checkout remained at `842a34130e82f241330707118a05bf6ed01e263e`, tree
`3fd4c4538d15b85233b1f8b582dce0f185dedba2`, on
`scale/generation-min-snr-matched-pilot-v1-20260825`, tracked-clean.

## Source hashes

```text
a4123cca01203a439378fb860914f021ad1f0ee762732dbe465e3c417b3fda5c  docs/records/2026-08-25_generation_min_snr_terminal_physical_replay_guard.md
6feeaea7909dc55cf0da44c25f65b75e032419404bfefeb1b2ae150d2fa6f2df  scripts/build_generation_min_snr_terminal_physical_guard.py
fc44914940664ee7f216b0732a8519dd24b76c0c6eacc254b7ad8db40159e0b0  scripts/verify_generation_min_snr_terminal_physical_guard.py
441cbcf297525887853cfc515af25cf916d0547bd900472341abd70e9c9cb4d9  src/cofitok/generation/min_snr_terminal_guard.py
bc6c6aa2909a07f21fe219f02811f642e6b263aedb6d5d707b7eeeca4d6840a0  tests/test_generation_min_snr_terminal_physical_guard.py
```

## Remote checks

The isolated checkout ran with `CUDA_VISIBLE_DEVICES=""`,
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `nice=19`, and `ionice -c 3`.

- focused and related guard/protocol tests: `57 passed`
- Python compilation: passed
- builder and verifier `--help`: passed with the isolated checkout on `PYTHONPATH`
- `git diff --check`: passed
- live preparation replay: passed; preparation SHA256
  `8fa3b768015c08c6f34627dc767c4b2fddd465a43913495120a496ef411c6830`

No terminal guard was built or verified because the four 10K terminal sampling
arms do not exist yet.

## Live-chain boundary at verification

- pilot phase: `training_cofitok`
- latest observed pilot step: `4150`
- latest observed samples: `265600`
- metrics: `84` rows, finite, strictly increasing, and
  `samples_seen == step * 64`
- monitor: `running`, `issues=[]`, permanently non-authorizing
- GPU: one direct pilot trainer only (`PID 407563` at the snapshot)
- free storage: `263921410048` bytes
- `terminal_status=hold`
- `generation_advantage_proven=false`
- training, sampling, evaluation, continuation, full-300K, promotion, export,
  release, and process-signal permissions: all `false`

## CoFiTok 5K physical checkpoint replay

The first pilot checkpoint was atomically published after the initial remote
verification. A second CUDA-hidden replay ran at `nice=19`, `ionice=idle`,
`OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`; the complete wrapper exited 0.

```text
checkpoint: checkpoint_step_00005000.pt
checkpoint bytes: 1010933994
checkpoint SHA256: b6f25451d2a21eda42e879437f2c8d7101274f73dbba72a147af82c292f1985b
sidecar bytes: 599
sidecar SHA256: fa5670714e673de44541f332434395807c2d9b47fa1ddc841bce25322754dfa4
latest bytes: 669
latest SHA256: 58b4e1be3a2ee452fc6e229bd88d7849ed58f0b8b3670183eac0b981949c70f2
```

The standard checkpoint verifier physically rehashed the payload. The
payload, sidecar, and latest identities remained unchanged across the replay.
Git revision/branch/clean state, dataset identity, runtime identity, checkpoint
format, step, filenames, and all latest fields matched. The 101-row metrics
prefix was finite and strictly increasing, satisfied
`samples_seen == step * 64`, and kept `epsilon <= epsilon_unweighted` on every
row. The step-5K row bound 320,000 samples and fixed-validation event 4 using
noise seed 102030 and 64 images. The health monitor remained `running` with
`issues=[]`; the sole GPU process remained the original trainer.

## Read-only early-warning alignment

At the live pilot prefix (steps 1K through 5K), the fixed-validation
epsilon MSE was compared with the frozen gamma=0 controls. This is an
exploratory same-step warning only; it is not a terminal quality result and
does not authorize a route change.

```text
step,pilot,legacy_cofitok,legacy_dense,pilot_vs_legacy_percent,pilot_vs_dense_percent
1000,0.045927882195,0.042286030948,0.042490236461,8.612422,8.090437
2000,0.037903025746,0.032742884010,0.032447181642,15.759582,16.814539
3000,0.043239824474,0.038105003536,0.038265690207,13.475451,12.998940
4000,0.034968268126,0.029629319906,0.029689561576,18.019139,17.779672
5000,0.036454834044,0.031380798668,0.031351022422,16.169236,16.279570
```

The five-event mean is `0.039698766917` for the pilot versus
`0.034828807414` for legacy CoFiTok (`+13.982562%`) and
`0.034848738462` for legacy dense (`+13.917372%`). The pilot is lower in
`0/5` events against either control. The signal is retained as a warning only.
The pilot must still reach its source-bound 50K physical audit and all four
terminal arms; `generation_advantage_proven` remains false.

The comparison re-read exact JSONL prefixes through step 5K. Each prefix has
101 rows:

```text
pilot CoFiTok: 104003 bytes, cd2dfcb6a61f89f29bde5e325fb4f34505e6375af57d13d370d00337e5ce2b1d
legacy CoFiTok: 95262 bytes, 290b77eae0d70ae3b05642826c92f15a915ec7256d473a5250a1352136035a4d
legacy dense: 88723 bytes, 31f58e2b02c6a6350ee3b3adbed7bc777bbeeda6dd207aa1fd7d397a6e3a3dc7
```

All three validation streams use event indices `0..4`, batch indices `0..4`,
noise seed `102030`, and 64 images per event at steps 1K through 5K.
