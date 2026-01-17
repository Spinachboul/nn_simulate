from core.models import ModelConfig, TrainingResult
from core.rules import *
from core.utils import clamp


class TrainingSimulator:

    def run(self, config: ModelConfig) -> TrainingResult:
        reasons = []

        accuracy = 0.5
        stability = 0.5
        speed = 0.5
        overfit = 0.0

        # ---- Loss vs Task ----
        loss_effect = LOSS_TASK_RULES.get(config.task, {}).get(config.loss, -1.0)
        if loss_effect <= -1.0:
            return TrainingResult(
                accuracy=0.0,
                stability=0.0,
                convergence_speed=0.0,
                overfitting_risk=1.0,
                training_time_ms=0,
                failed=True,
                reasons=["Loss function incompatible with task"],
            )

        accuracy += loss_effect
        reasons.append("Loss function influenced task performance")

        # ---- Activation vs Depth ----
        act_rules = ACTIVATION_RULES.get(config.activation, {})
        depth_effects = act_rules.get(config.depth, {})

        speed += depth_effects.get("speed", 0.0)
        stability += depth_effects.get("stability", 0.0)

        if depth_effects:
            reasons.append("Activation-depth interaction affected gradients")

        # ---- Normalization ----
        norm_effects = NORMALIZATION_RULES.get(config.normalization, {})
        speed += norm_effects.get("speed", 0.0)
        stability += norm_effects.get("stability", 0.0)

        if norm_effects:
            reasons.append("Normalization improved training dynamics")

        # ---- Overfitting ----
        overfit += OVERFITTING_RULES.get(config.depth, 0.0)
        if overfit > 0.3:
            accuracy -= 0.2
            reasons.append("Model overfit due to excessive depth")

        # ---- Clamp metrics ----
        accuracy = clamp(accuracy)
        stability = clamp(stability)
        speed = clamp(speed)
        overfit = clamp(overfit)

        training_time = int(2000 * (1.2 - speed))

        return TrainingResult(
            accuracy=accuracy,
            stability=stability,
            convergence_speed=speed,
            overfitting_risk=overfit,
            training_time_ms=training_time,
            failed=False,
            reasons=reasons,
        )
