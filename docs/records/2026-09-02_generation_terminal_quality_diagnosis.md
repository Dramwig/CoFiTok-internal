# Terminal quality diagnosis (2026-09-02)

This is a CPU-only, non-authorizing analysis of the already completed matched
100K terminal reports.  It does not sample, train, select a candidate, or
change an execution gate.

## Evidence

The terminal protocol used 10,000 generated images per method, 50,000 real
images, DDIM-100, CFG 1.5, bf16, and the shared balanced-modulo class schedule.
The source reports and hashes are recorded in
`artifacts/reports/generation/terminal_quality_diagnosis_2026-09-02.json`.

| metric | CoFiTok K8 | dense identity | interpretation |
| --- | ---: | ---: | --- |
| FID | 115.2622 | 123.0210 | CoFiTok lower at this terminal snapshot |
| precision | 0.7556 | 0.6653 | CoFiTok sharper |
| recall | 0.00832 | 0.01000 | both have severe support collapse |
| class top-1 | 0.24% | 0.17% | both near random 1000-way scale |
| predicted-class coverage | 73.4% | 69.6% | CoFiTok somewhat broader, still incomplete |
| validation epsilon MSE | 0.0292560 | 0.0292407 | essentially matched |

The 100K milestone is an important counter-signal: CoFiTok FID `219.86`
versus dense `134.68`, with a named quality alert.  Therefore the terminal
FID/precision difference cannot by itself establish a robust generation
advantage.

## Diagnosis

The evidence points primarily to distribution support and generation
stability/exposure or capacity, not a class-only failure and not a large
primary epsilon-loss divergence.  The mechanism evidence remains intact:
ordered-prefix rank, zero-token, shuffle mismatch, and coarse-token
utilization pass.

The existing 1,000-sample recovery screen and Min-SNR screen found no shared
repair candidate.  If a future exact GPU stage is authorized, the current
evidence better motivates an exposure/capacity qualification than a
conditioning-only or Min-SNR intervention.  No arm is selected here, and
`generation_advantage_proven` remains `false`.
