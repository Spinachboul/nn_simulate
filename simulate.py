import yaml
from core.simulator import TrainingSimulator
from core.models import ModelConfig
from core.validation import validate_config_dict, ValidationError
from core.scoring import compute_score


def load_config(path="model.yaml") -> ModelConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    validate_config_dict(data)

    return ModelConfig(
        task=data["task"],
        activation=data["activation"],
        loss=data["loss"],
        normalization=data["normalization"],
        depth=data["depth"],
    )


def main():
    print("=" * 48)
    print(" Neural Network Intuition Simulator ")
    print("=" * 48)

    try:
        config = load_config()
    except ValidationError as e:
        print("\n❌ CONFIGURATION ERROR\n")
        print(str(e))
        exit(1)

    simulator = TrainingSimulator()
    result = simulator.run(config)
    score = compute_score(result)

    status = "FAILED" if result.failed else "COMPLETE"

    print(f"\n🔎 TRAINING {status}\n")

    print("Configuration:")
    print(f"- Task: {config.task}")
    print(f"- Activation: {config.activation}")
    print(f"- Loss: {config.loss}")
    print(f"- Normalization: {config.normalization}")
    print(f"- Depth: {config.depth}")

    print("\nMetrics:")
    print(f"- Accuracy: {result.accuracy:.2f}")
    print(f"- Stability: {result.stability:.2f}")
    print(f"- Speed: {result.convergence_speed:.2f}")
    print(f"- Overfitting: {result.overfitting_risk:.2f}")
    print(f"- Score: {score}/100")

    print("\nPost-mortem:")
    for r in result.reasons:
        print(f"- {r}")

    write_report(config, result, score)


def write_report(config, result, score, path="reports/latest.md"):
    import os
    os.makedirs("reports", exist_ok=True)

    verdict = "❌ Training Failed" if result.failed else "✅ Training Complete"

    with open(path, "w") as f:
        f.write(f"## {verdict}\n\n")

        f.write("### Score\n")
        f.write(f"**{score} / 100**\n\n")

        f.write("### Configuration\n")
        for k, v in config.__dict__.items():
            f.write(f"- {k}: {v}\n")

        f.write("\n### Metrics\n")
        f.write(f"- Accuracy: {result.accuracy:.2f}\n")
        f.write(f"- Stability: {result.stability:.2f}\n")
        f.write(f"- Training Speed: {result.convergence_speed:.2f}\n")
        f.write(f"- Overfitting Risk: {result.overfitting_risk:.2f}\n")

        f.write("\n### Post-mortem\n")
        for r in result.reasons:
            f.write(f"- {r}\n")


if __name__ == "__main__":
    main()
