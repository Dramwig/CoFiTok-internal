# Full-run checkpoint cadence

Date: 2026-07-12

Branch: `scale/generative-system`

The matched full ImageNet-256 300K configurations now save every 5,000
optimizer steps rather than every 10,000. At the observed 10% CoFiTok rate of
approximately 2.39 seconds per optimizer step, this reduces the nominal
recovery-point interval from about 6.6 hours to about 3.3 hours.

Both CoFiTok and `dense_identity` use the same cadence and retain the latest
three checkpoints. This changes only checkpoint I/O frequency; data order,
optimizer updates, LR schedule, EMA, model capacity, and evaluation protocol
remain matched.

The active 10% pair already uses a 5,000-step cadence and remains pinned to
commit `781a014`. The full-run configuration change will take effect only after
the 10% promotion gate passes.
