from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ModelConfig:
    task: str
    activation: str
    loss: str
    normalization: str
    depth: str


@dataclass
class TrainingResult:
    accuracy: float
    stability: float
    convergence_speed: float
    overfitting_risk: float
    training_time_ms: int
    failed: bool
    reasons: List[str]
