import os
from openai import OpenAI
from dataset import load_tau_dataset
from graph import build_workflow
from models import GraphState


def main():
    print("Конфигурация API")
    api_key = "sk-2MRsCHDx6oRaDRSHQV0W3rsAB4OyDz0037WZx4ytbpeFFuYg"
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    base_url = "https://api.weegam.com/uni/v1/"
    model_name = "gemini-3-flash-preview"

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)

    print("Настройка пайплайна")

    default_task = "Оцени диалог. 1 - агент решил задачу, 0 - не решил."
    user_task = input(f"Описание задачи (Enter для '{default_task}'): ").strip()
    task_desc = user_task if user_task else default_task

    num_train = int(input("Размер обучающей выборки (Train) [10]: ") or 10)
    num_val = int(input("Размер валидационной выборки (Val) [20]: ") or 20)
    max_iter = int(input("Максимальное количество итераций [3]: ") or 3)
    target_acc = float(input("Целевая точность в %[90.0]: ") or 90.0)

    train_data, val_data = load_tau_dataset(num_train, num_val)

    initial_state = GraphState(
        task_description=task_desc,
        train_examples=train_data,
        val_examples=val_data,
        current_prompt="",
        feedback="",
        iteration=1,
        max_iterations=max_iter,
        target_accuracy=target_acc,
        best_prompt="",
        best_accuracy=0.0
    )

    graph = build_workflow(client, model_name)
    final_state = graph.invoke(initial_state)

    print("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
    print(f"Лучшая точность: {final_state['best_accuracy']:.1f}%")

    with open("best_prompt.txt", "w", encoding="utf-8") as f:
        f.write(final_state['best_prompt'])
    print("Лучший промпт сохранен в 'best_prompt.txt'.")


if __name__ == "__main__":
    main()