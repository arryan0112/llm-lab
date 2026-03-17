import yaml
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ExperimentConfig:
    name: str
    model: str
    method: str
    dataset: str
    task: str
    num_samples: int
    epochs: int
    batch_size: int
    learning_rate: float
    max_length: int
    output_dir: str
    lora_rank: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None
    target_modules: Optional[List[str]] = None

def load_config(config_path: str = "configs/experiments.yaml") -> List[ExperimentConfig]:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    return [ExperimentConfig(**exp) for exp in raw["experiments"]]

def load_single_config(name: str, config_path: str = "configs/experiments.yaml") -> ExperimentConfig:
    for c in load_config(config_path):
        if c.name == name:
            return c
    raise ValueError(f"No experiment named '{name}' found.")

if __name__ == "__main__":
    configs = load_config()
    for c in configs:
        print(f"OK: {c.name} | method={c.method} | rank={c.lora_rank}")