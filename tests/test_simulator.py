from core.simulator import TrainingSimulator
from core.models import ModelConfig
from core.constants import *

def test_invalid_loss_fails():
    sim = TrainingSimulator()
    config = ModelConfig(
        task=TASK_REGRESSION,
        activation=ACT_RELU,
        loss=LOSS_BCE,
        normalization=NORM_NONE,
        depth=DEPTH_MEDIUM,
    )
    result = sim.run(config)
    assert result.failed is True
