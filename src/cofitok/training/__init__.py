from cofitok.training.losses import LossBreakdown, compute_losses
from cofitok.training.step import SmokeStepOutput, run_smoke_step
from cofitok.training.checkpointing import (
    backfill_training_checkpoint_integrity,
    load_training_checkpoint,
    save_training_checkpoint,
)
from cofitok.training.ema import ExponentialMovingAverage
from cofitok.training.conditioning import (
    ClassConditioningRankingResult,
    ClassConditioningResidualAlignmentResult,
    class_conditioning_ranking_loss,
    class_conditioning_residual_alignment_loss,
)
from cofitok.training.authorization import (
    build_generation_training_authorization,
    capture_generation_training_authorization,
    validate_checkpoint_training_authorization,
    validate_generation_training_authorization,
)
from cofitok.training.metrics import (
    ensure_fresh_training_output,
    reconcile_metrics_for_resume,
)
from cofitok.training.rollout import (
    consistency_weight_scale,
    ema_teacher_consistency_loss,
    one_step_rollout_consistency_loss,
    rollout_consistency_loss,
    rollout_consistency_weight_scale,
)

__all__ = [
    "ExponentialMovingAverage",
    "ClassConditioningRankingResult",
    "ClassConditioningResidualAlignmentResult",
    "LossBreakdown",
    "SmokeStepOutput",
    "compute_losses",
    "class_conditioning_ranking_loss",
    "class_conditioning_residual_alignment_loss",
    "build_generation_training_authorization",
    "capture_generation_training_authorization",
    "ensure_fresh_training_output",
    "backfill_training_checkpoint_integrity",
    "load_training_checkpoint",
    "consistency_weight_scale",
    "ema_teacher_consistency_loss",
    "one_step_rollout_consistency_loss",
    "rollout_consistency_loss",
    "reconcile_metrics_for_resume",
    "rollout_consistency_weight_scale",
    "run_smoke_step",
    "save_training_checkpoint",
    "validate_generation_training_authorization",
    "validate_checkpoint_training_authorization",
]
