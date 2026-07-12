# Absolute quality gate for the full generation system

Date: 2026-07-12

Branch: `scale/generative-system`

The generation gate now distinguishes the 10% scaling decision from final
large-scale readiness. A passing scaling report authorizes full ImageNet-256
training with `promote_to_full_imagenet256`; a passing full report emits
`large_scale_generation_ready`.

The full report is not allowed to pass on relative parity with a weak dense
control alone. Before full training results exist, the formal runbook commits
to all of the following:

- 50,000 generated samples per matched method;
- finite FID, Inception Score mean/std, precision, and recall for both methods;
- CoFiTok FID no more than 5% worse than the matched dense control;
- CoFiTok absolute FID at most 20.0;
- endpoint clean MSE no more than 5% worse than dense;
- ordered prefix rank 1, exact zero-token output, and shuffle mismatch;
- matching EMA checkpoint hashes and sampling provenance.

The 10% scaling gate uses a deliberately loose absolute FID ceiling of 100.0
because it trains on only 10% of ImageNet. It remains a go/no-go architecture
screen and cannot declare the large-scale system complete.
