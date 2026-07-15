# Generation Bundle Fetch Fix

Date: 2026-07-15

The immutable legacy ImageNet-256 10% matched pair completed at 50,000 steps
for both CoFiTok and `dense_identity`. The one-shot deployment waiter then
failed before moving the formal repository from pinned revision `781a014`.
Its log recorded:

```text
fatal: couldn't find remote ref HEAD
```

The prerequisite-aware bundle advertised only
`refs/heads/scale/generative-system`; it did not advertise a synthetic `HEAD`
ref. The deployment helper incorrectly requested `HEAD` from the bundle.

The helper now independently requires exactly one advertised head, binds its
object ID to the requested target revision, permits only the two forms produced
by the supported bundle creators (`HEAD` or
`refs/heads/scale/generative-system`), and fetches the actual advertised ref
into `FETCH_HEAD`. Real-Git regression tests create both incremental bundle
forms from a pinned parent, fetch each advertised head into an isolated pinned
checkout, and verify that `FETCH_HEAD` is the target object.

No legacy checkpoint, metric log, training report, or locked paper artifact was
modified by the failed deployment. The formal remote HEAD remained at
`781a01444fddbf0d48a427ba58bdeed50167b5be`, and no deployment receipt or
completion supervisor existed before the corrected transition was rehearsed.
