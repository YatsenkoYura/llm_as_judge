import pandas as pd
from models import UniversalExample


def load_dices_dataset(file_path: str, num_train: int, num_val: int):
    df = pd.read_csv(file_path)

    unique_dialogs = df.drop_duplicates(subset="item_id")

    formatted_examples = []

    for _, row in unique_dialogs.iterrows():
        context_str = f"ИСТОРИЯ ДИАЛОГА\n{row['context']}\n\nПОСЛЕДНИЙ ОТВЕТ ИИ \n{row['response']}"

        gold_label = str(row['safety_gold']).strip().lower()
        if gold_label == 'yes':
            label_val = "0"
        else:
            label_val = "1"

        explanation = (
            f"Степень вреда: {row['degree_of_harm']}. "
            f"Категория: {row['harm_type']}. "
            f"Обоснование экспертов: {row['safety_gold_reason']}"
        )

        # Упаковываем в наш универсальный формат
        example = UniversalExample(
            context=context_str,
            expected_label=label_val,
            explanation=explanation
        )
        formatted_examples.append(example)

    train_data = formatted_examples[:num_train]
    val_data = formatted_examples[num_train: num_train + num_val]

    return train_data, val_data