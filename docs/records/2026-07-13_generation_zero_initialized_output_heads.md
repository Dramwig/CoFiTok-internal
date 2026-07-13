# Zero-initialized generation output heads (2026-07-13)

## Problem

The scalable U-Net already zero initialized residual output convolutions and
attention projections, but its final token heads used PyTorch's default random
initialization. Fresh CoFiTok training therefore summed eight random synthesized
components while the matched dense control emitted one random epsilon field.
That method-dependent initial output scale was unnecessary for diffusion
optimization and weakened the direct-pair initialization contract.

## Implementation

Every final token-head weight and bias in `ScalableUNetTokenPredictor` is now
initialized to zero. This applies uniformly to restricted K-token CoFiTok and
the one-head `dense_identity` control. The restricted synthesis projections are
not zeroed: at the initial zero-token state they preserve a nonzero derivative
from epsilon loss to each token head, allowing the first optimizer step to leave
zero output.

Existing checkpoints are unaffected because strict state loading replaces the
freshly initialized heads. The active 10% queue remains on its pinned revision;
the change applies to new full ImageNet-256 training after controlled deployment.

## Verification

Tests require both production method forms to emit exact zero epsilon before
training, produce nonzero token-head gradients on the first backward pass, and
emit nonzero epsilon after one AdamW update. The class/time-conditioning test
also explicitly activates a token head and continues to prove that conditioning
terminates inside `T_k` while the restricted `S_k` receives tokens only.
