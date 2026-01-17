from core.constants import *

ALLOWED = {
    "task": {TASK_BINARY, TASK_MULTICLASS, TASK_REGRESSION},
    "activation": {ACT_RELU, ACT_SIGMOID, ACT_TANH},
    "loss": {LOSS_BCE, LOSS_CCE, LOSS_MSE},
    "normalization": {NORM_NONE, NORM_BATCH},
    "depth": {DEPTH_SHALLOW, DEPTH_MEDIUM, DEPTH_DEEP},
}


class ValidationError(Exception):
    pass


# vaidates the configuration dictionary
def validate_config_dict(data: dict):
    # check for any missing keys
    for key in ALLOWED.keys():
        if key not in data:
            raise ValidationError(f"Missing key: {key}")
        if data[key] not in ALLOWED[key]:
            raise ValidationError(
                f"Invalid value for {key}: {data[key]}. Allowed values are: {ALLOWED[key]}"
            )

    return True

