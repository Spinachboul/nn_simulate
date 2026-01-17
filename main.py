from fastapi import FastAPI
from core.models import ModelConfig
from core.simulator import TrainingSimulator

app = FastAPI()
simulator = TrainingSimulator()


@app.post("/train")
def train(config: ModelConfig):
    result = simulator.run(config)
    return result
