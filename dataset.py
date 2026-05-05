import json
from models import UniversalExample


def load_tau_dataset(num_train, num_val):
    file_path = "/home/kano/.cache/huggingface/hub/datasets--cmu-lti--tau-usi/snapshots/400222f04769da7b6c22d230d3cea98084b1b659/data/tau_bench_tasks_unified.json"

    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        all_items = []
        for task_name, task_data in raw_data.items():
            if isinstance(task_data, list):
                all_items.extend(task_data)
            else:
                all_items.append(task_data)
    else:
        all_items = raw_data

    formatted_examples = []
    for item in all_items:
        context_str = json.dumps(item, ensure_ascii=False, indent=2)

        label_val = str(item.get("reward", item.get("success", "1")))

        example = UniversalExample(
            context=context_str,
            expected_label=label_val,
            explanation="Оценка извлечена из датасета TAU"
        )
        formatted_examples.append(example)

    train_data = formatted_examples[:num_train]
    val_data = formatted_examples[num_train: num_train + num_val]

    return train_data, val_data