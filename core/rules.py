from core.constants import *

LOSS_TASK_RULES = {
    TASK_BINARY: {
        LOSS_BCE: +0.3,
        LOSS_MSE: -0.25,
        LOSS_CCE: -1.0,
    },
    TASK_MULTICLASS: {
        LOSS_CCE: +0.3,
        LOSS_MSE: -0.4,
        LOSS_BCE: -1.0,
    },
    TASK_REGRESSION: {
        LOSS_MSE: +0.3,
        LOSS_BCE: -1.0,
        LOSS_CCE: -1.0,
    },
}

ACTIVATION_RULES = {
    ACT_RELU: {
        DEPTH_DEEP: {"speed": +0.2, "stability": -0.3},
        DEPTH_MEDIUM: {"speed": +0.1},
    },
    ACT_SIGMOID: {
        DEPTH_DEEP: {"speed": -0.4},
        DEPTH_SHALLOW: {"stability": +0.1},
    },
    ACT_TANH: {
        DEPTH_MEDIUM: {"stability": +0.1},
    },
}

NORMALIZATION_RULES = {
    NORM_NONE: {},
    NORM_BATCH: {
        "stability": +0.25,
        "speed": +0.1,
    },
}

OVERFITTING_RULES = {
    DEPTH_SHALLOW: -0.2,
    DEPTH_MEDIUM: 0.0,
    DEPTH_DEEP: +0.4,
}
