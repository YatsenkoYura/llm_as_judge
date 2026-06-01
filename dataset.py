import pandas as pd
import random
from models import UniversalExample
from typing import List, Tuple


def process_dataset(
    df: pd.DataFrame,
    context_col: str,
    label_cols: List[str],
    explanation_col: str,
    num_train: int,
    num_val: int,
    val_for_1: List[str] = None,
    val_for_0: List[str] = None,
    case_sensitive: bool = False,
    shuffle: bool = True,
    shuffle_seed: int = 42
) -> Tuple[List[UniversalExample], List[UniversalExample]]:
    if val_for_1 is None:
        val_for_1 = ['yes', '1', 'true', '1.0']
    if val_for_0 is None:
        val_for_0 = ['no', '0', 'false', '0.0']

    if not case_sensitive:
        val_for_1 = [str(v).strip().lower() for v in val_for_1]
        val_for_0 = [str(v).strip().lower() for v in val_for_0]
    else:
        val_for_1 = [str(v).strip() for v in val_for_1]
        val_for_0 = [str(v).strip() for v in val_for_0]

    formatted_examples = []

    for _, row in df.iterrows():
        context_str = str(row[context_col]) if pd.notna(row[context_col]) else ""
        explanation_str = str(row[explanation_col]) if pd.notna(row[explanation_col]) else ""
        
        label_val = "0"
        has_one = False
        all_zero = True
        
        for col in label_cols:
            val = str(row[col]) if pd.notna(row[col]) else ""
            val = val.strip()
            if not case_sensitive:
                val = val.lower()
                
            if val in val_for_1:
                has_one = True
                break
            elif val not in val_for_0:
                all_zero = False
                
        if has_one:
            label_val = "1"
        elif all_zero:
            label_val = "0"
        else:
            label_val = "unknown"

        if label_val in ["0", "1"]:
            example = UniversalExample(
                context=context_str,
                expected_label=label_val,
                explanation=explanation_str
            )
            formatted_examples.append(example)

    if shuffle:
        random.seed(shuffle_seed)
        random.shuffle(formatted_examples)

    train_data = formatted_examples[:num_train]
    val_data = formatted_examples[num_train: num_train + num_val]

    return train_data, val_data