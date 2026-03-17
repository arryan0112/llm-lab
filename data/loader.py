from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import List, Dict

# ── Prompt templates ───────────────────────────────────────────────

def format_instruction(sample: dict) -> str:
    if sample.get("input", "").strip():
        return (
            f"### Instruction:\n{sample['instruction']}\n\n"
            f"### Input:\n{sample['input']}\n\n"
            f"### Response:\n{sample['output']}"
        )
    return (
        f"### Instruction:\n{sample['instruction']}\n\n"
        f"### Response:\n{sample['output']}"
    )

def format_qa(sample: dict) -> str:
    answer = sample['answers']['text'][0] if sample['answers']['text'] else "No answer"
    return (
        f"### Context:\n{sample['context']}\n\n"
        f"### Question:\n{sample['question']}\n\n"
        f"### Answer:\n{answer}"
    )

def format_summarization(sample: dict) -> str:
    return (
        f"### Article:\n{sample['article'][:1500]}\n\n"
        f"### Summary:\n{sample['highlights']}"
    )

# ── Dataset loaders ────────────────────────────────────────────────

def load_alpaca(num_samples: int) -> List[Dict]:
    print(f"  Loading Alpaca ({num_samples} samples)...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.select(range(min(num_samples, len(ds))))
    return [{"text": format_instruction(s), "task": "instruction"} for s in ds]

def load_squad(num_samples: int) -> List[Dict]:
    print(f"  Loading SQuAD ({num_samples} samples)...")
    ds = load_dataset("squad", split="validation")
    ds = ds.select(range(min(num_samples, len(ds))))
    return [{"text": format_qa(s), "task": "qa",
             "reference": s['answers']['text'][0]} for s in ds]

def load_cnn(num_samples: int) -> List[Dict]:
    print(f"  Loading CNN/DailyMail ({num_samples} samples)...")
    ds = load_dataset("cnn_dailymail", "3.0.0", split="validation")
    ds = ds.select(range(min(num_samples, len(ds))))
    return [{"text": format_summarization(s), "task": "summarization",
             "reference": s['highlights']} for s in ds]

def load_samples(dataset_name: str, num_samples: int) -> List[Dict]:
    loaders = {"alpaca": load_alpaca, "squad": load_squad, "cnn": load_cnn}
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose from {list(loaders.keys())}")
    return loaders[dataset_name](num_samples)

# ── PyTorch Dataset wrapper ────────────────────────────────────────

class LLMDataset(Dataset):
    def __init__(self, samples: List[Dict], tokenizer: PreTrainedTokenizer, max_length: int):
        self.encodings = []
        print(f"  Tokenizing {len(samples)} samples...")
        for sample in samples:
            enc = tokenizer(
                sample["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )
            enc["labels"] = enc["input_ids"].clone()
            self.encodings.append({k: v.squeeze(0) for k, v in enc.items()})

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]