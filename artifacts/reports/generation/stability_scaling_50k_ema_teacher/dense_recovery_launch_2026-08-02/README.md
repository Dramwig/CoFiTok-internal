# Stability dense-identity 50K recovery launch evidence

This pack records the authorized dense-only recovery launch after the completed
CoFiTok stability member. It contains only small reports and initial metrics;
no checkpoint payload, sample tree, feature cache, or long log is copied.

The launch independently revalidated the clean controller and immutable
training checkouts, the completed CoFiTok step-50K checkpoint payload and
integrity sidecar, frozen `64x1` runtime, exact matched configs, deployment
receipt, exclusive GPU availability, and storage runway. The recovery runbook
then repeated those checks under its non-blocking controller lock before
starting only the dense member.

At the authoritative monitor refresh, dense training was healthy at step 100
with 6,400 images seen. The root recovery controller, pair monitor, watchdog,
GPU trainer leader, post-evaluation waiter, and readiness waiter each had one
live identity-bound process. DataLoader children inherited the trainer argv but
did not own GPU compute contexts. The post-evaluation waiter was waiting for the
completed pair; the receipt-bound readiness waiter was waiting for a passing
stability gate and retained `full_training_launch_allowed=false`.

This launch does not authorize full 300K training. `manifest.json` binds every
small evidence file by byte count and SHA256.
