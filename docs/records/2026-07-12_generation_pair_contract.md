# Matched generation pair contract

Date: 2026-07-12

Branch: `scale/generative-system`

Parameter proximity and equal optimizer settings are insufficient to establish
a direct factorization baseline. A class-dropout, backbone-width, attention, or
checkpointing drift could change generation quality while still passing the old
top-level section comparison.

`cofitok.generation_pair.generation_pair_contract` now provides one shared
contract for `validate_generation_training_pair.py` and
`build_generation_gate_report.py`. It requires:

- exact equality of resolved data, diffusion, runtime, and optimization sections;
- exact equality of every shared model field, including image shape, U-Net
  width/depth/channel multipliers, attention, dropout, gradient checkpointing,
  class count, and class-dropout probability;
- CoFiTok `token_count>1`, feedback enabled, and restricted synthesis;
- dense `token_count=1`, feedback disabled, and `dense_identity` synthesis;
- equal positive primary epsilon-loss weight;
- every dense auxiliary `*_weight` equal to zero;
- the existing <=2% total parameter-gap constraint.

Only token count/channels, feedback, synthesis implementation, synthesis
channel/stride/gamma fields, and CoFiTok factorization auxiliaries are permitted
to differ. Tests resolve both checked-in 10% and full configurations through the
real config loader before validating them, so defaults omitted from JSON are
also covered. Negative tests change shared class dropout and dense auxiliary
losses and require both the pre-deployment validator and scientific gate to hold.
