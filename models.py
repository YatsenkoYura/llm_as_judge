from pydantic import BaseModel
from typing import List, TypedDict, Literal

class UniversalExample(BaseModel):
    context: str
    expected_label: str
    explanation: str

class JudgeResponse(BaseModel):
    user_request: str
    system_actions: str
    agent_words: str
    verification: str
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