from pydantic import BaseModel
from typing import List, TypedDict, Literal
from dataclasses import dataclass


class UniversalExample(BaseModel):
    context: str
    expected_label: str
    explanation: str


class JudgeResponse(BaseModel):
    analysis: str
    evidence: str
    label: Literal["0", "1"]


class GraphState(TypedDict):
    task_description: str
    train_examples: List[UniversalExample]
    val_examples: List[UniversalExample]
    current_prompt: str
    feedback: str
    iteration: int
    max_iterations: int
    target_accuracy: float
    best_prompt: str
    best_accuracy: float
    metrics_history: List[dict]


@dataclass
class RunConfig:
    """централизованный конфиг запуска."""
    api_token: str = ""
    api_base_url: str = "https://api.openai.com/v1"
    evaluator_model: str = "gemini-3-flash-preview"
    judge_model: str = "gemini-3-flash-preview"
    task_description: str = ""
    num_train: int = 10
    num_val: int = 20
    max_iterations: int = 3
    target_accuracy: float = 90.0
    use_shuffle: bool = True
    shuffle_seed: int = 42
    use_smart_truncation: bool = True
    truncation_max_len: int = 1000
    use_balanced_accuracy: bool = True
    max_parallel_workers: int = 5
    max_retries: int = 3