# Matched 1K rollout-stability qualification

Status: `pass`

The accepted probe combines:

- RGB tail-3 fixed bases:
  channels `[4,4,8,8,8,1,1,1]`,
  strides `[16,16,8,8,4,1,1,1]`.
- Capacity-blended stable-Hellinger path energy.
- Bounded low-SNR high-frequency loss.
- One detached generated-state step followed by a bounded
  `clipped_x0` consistency loss.

CoFiTok and dense used the same data, backbone, seed, runtime, optimizer,
training horizon, and rollout-consistency settings. The raw model checkpoint
is authoritative for this short-run qualification; EMA is retained as an
auxiliary lag diagnostic.

| gate | result | limit |
|---|---:|---:|
| tail-two energy | 0.570061 | <= 0.65 |
| maximum single-token energy | 0.286358 | <= 0.35 |
| ordered path rank | 1 | = 1 |
| endpoint CoFiTok/dense | 1.013961 | <= 1.05 |
| validation CoFiTok/dense | 1.011818 | <= 1.05 |
| peak predicted-x0 HF CoFiTok/dense | 1.453409 | <= 1.50 |
| reconstruction CoFiTok/dense | 0.977907 | <= 1.05 |
| zero-token max abs | 0 | <= 1e-8 |
| shuffle/ordered endpoint | 103.367x | >= 2x |

The four predicted-x0 high-frequency ratios at DDIM timesteps
`595/394/192/91` are `1.2150/1.4534/1.3863/1.3512`. The corresponding formal
v3 maximum was `10.032x`.

Authoritative report:

```text
qualification_report.json
sha256: 225839d69a239483852b40413fce17e0ff56467c6be3bf3d2c5a4b26228715ef
```

This result authorizes a matched 5K scaling gate. It does not authorize a new
formal 50K run, full ImageNet-256 300K training, or a generation-quality claim.
