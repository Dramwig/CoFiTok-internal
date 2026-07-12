# CoFiTok Idea Requirements Matrix

Status: `ready`

| requirement | group | status | evidence | next action |
|---|---|---|---|---|
| `dense_noise_factorization_code` | method | `ok` | path=C:/Users/zixi-/Desktop/PaperField/CoFiTok/CoFiTok-internal/src/cofitok/models/cofitok.py<br>required_snippet_count=7<br>present_count=7 | Keep as invariant. |
| `restricted_synthesis_operator_code` | method | `ok` | path=C:/Users/zixi-/Desktop/PaperField/CoFiTok/CoFiTok-internal/src/cofitok/models/synthesis.py<br>required_snippet_count=6<br>present_count=6 | Keep as invariant. |
| `losses_and_diagnostics_code` | method | `ok` | losses=C:/Users/zixi-/Desktop/PaperField/CoFiTok/CoFiTok-internal/src/cofitok/training/losses.py<br>diagnostics=C:/Users/zixi-/Desktop/PaperField/CoFiTok/CoFiTok-internal/src/cofitok/diagnostics.py<br>required_snippet_count=9<br>present_count=9 | Keep as invariant. |
| `multi_dataset_coverage` | evidence | `ok` | see JSON | Keep as invariant. |
| `prefix_denoising_evidence` | evidence | `ok` | see JSON | Keep as invariant. |
| `zero_random_shuffle_diagnostics` | evidence | `ok` | see JSON | Keep as invariant. |
| `order_ablation_and_token_scaling` | evidence | `ok` | see JSON | Keep as invariant. |
| `degeneration_and_predictor_ablations` | evidence | `ok` | see JSON | Keep as invariant. |
| `quality_and_sampling_evidence` | evidence | `ok` | quality_rows=58<br>sampling_rows=22<br>generated_quality_rows=29<br>quality_ok=True<br>sampling_ok=True | Keep as invariant. |
| `queued_publication_scale_batch` | publication | `ok` | run_count=10<br>quality_count=10<br>order_eval_count=24<br>sampling_count=6<br>generated_quality_count=6 | Run the queued batch on pro6000, then regenerate summaries and rerun publication readiness. |
| `publication_readiness` | publication | `ok` | max_quality_images=50000<br>max_generated_samples=50000<br>max_real_images=50000 | Keep as invariant. |
