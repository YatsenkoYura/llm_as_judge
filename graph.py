from langgraph.graph import StateGraph, END
from openai import OpenAI
from models import GraphState, JudgeResponse


def build_workflow(client: OpenAI, model_name: str):
    def generate_prompt_node(state: GraphState) -> dict:
        print(f"\n[Итерация {state['iteration']}] Генерация инструкции...")

        examples_blocks = []
        for ex in state['train_examples']:
            block = f"""
            Диалог/Логи: {ex.context[:700]}...
            Ожидаемая оценка: {ex.expected_label}
            Логика разметки: {ex.explanation}
            """
            examples_blocks.append(block.strip())

        examples_text = "\n\n".join(examples_blocks)

        system_message = f"""
        Ты эксперт по машинному обучению. Напиши системную инструкцию для LLM-судьи.
        Задача, которую решает оцениваемый агент: {state['task_description']}

        ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА ДЛЯ ТВОЕЙ ИНСТРУКЦИИ:
        1. Универсальная защита от лжи агента: ИИ-агенты часто галлюцинируют и уверенно заявляют пользователю, что "задача выполнена", даже если произошла системная ошибка или действие не было совершено. Запрети судье верить агенту на слово. Судья должен проверять успех ТОЛЬКО по фактическим системным вызовам (вызовам функций, логам инструментов).
        2. Краткость: Запрети судье лить воду. Рассуждения должны быть тезисными (буллиты).
        3. Few-Shot примеры: ОБЯЗАТЕЛЬНО включи прямо внутрь своей инструкции 2 коротких выдуманных примера оценки (один на 0, другой на 1), чтобы показать судье шаблон рассуждений.

        Исторические данные разметки для понимания задачи:
        {examples_text}
        """

        if state['iteration'] > 1 and state['current_prompt']:
            system_message += f"""
        Твоя ПРЕДЫДУЩАЯ версия инструкции:
        ---
        {state['current_prompt']}
        ---

        Она дала ошибки на валидации. Вот примеры того, где судья ошибся с этой инструкцией:
        {state['feedback']}

        Твоя задача: УЛУЧШИТЬ предыдущую инструкцию. 
        Добавь в нее новые правила, чтобы исправить эти ошибки, НО обязательно сохрани основной каркас, так как на многих примерах он уже работает правильно. Не начинай с нуля!
        Выведи только текст новой инструкции.
        """
        else:
            system_message += "\nВыведи только текст инструкции (без лишних приветствий)."

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": system_message}],
            temperature=0.4
        )

        generated_prompt = response.choices[0].message.content.strip()

        print("СГЕНЕРИРОВАННЫЙ ПРОМПТ:")
        print(generated_prompt if generated_prompt else "[ВНИМАНИЕ: ИИ ВЕРНУЛ ПУСТУЮ СТРОКУ!]")

        return {"current_prompt": generated_prompt}

    def evaluate_node(state: GraphState) -> dict:
        print(f"[Итерация {state['iteration']}] Оценка на валидационной выборке...")

        correct = 0
        errors = []
        total = len(state['val_examples'])

        for example in state['val_examples']:
            try:
                response = client.beta.chat.completions.parse(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": state['current_prompt']},
                        {"role": "user", "content": f"Оцени этот лог/диалог:\n{example.context}"}
                    ],
                    response_format=JudgeResponse,
                    temperature=0.0
                )
                result = response.choices[0].message.parsed

                # Приводим к float для надежного сравнения ("1.0" == "1")
                is_correct = float(result.label) == float(example.expected_label)

                print(f"Ожидали: {example.expected_label} | ИИ: {result.label} | Вердикт: {'✅' if is_correct else '❌'}")
                print(f"   Проверка: {result.verification[:80]}...")

                if is_correct:
                    correct += 1
                else:
                    errors.append(
                        f"Ожидали: {example.expected_label}, Получили: {result.label}\n"
                        f"Логика судьи (Проверка): {result.verification}"
                    )
            except Exception as e:
                print(f"❌ ОШИБКА API ИЛИ ПАРСИНГА: {e}")

        accuracy = (correct / total) * 100
        print(f"\nТочность: {accuracy:.1f}%")

        if errors:
            print(f"Пример ошибки из этой итерации:\n{errors[0]}")

        feedback = "\n\n".join(errors[:3]) if errors else ""

        updates = {
            "iteration": state["iteration"] + 1,
            "feedback": feedback
        }

        if accuracy > state["best_accuracy"]:
            updates["best_accuracy"] = accuracy
            updates["best_prompt"] = state["current_prompt"]

        return updates

    def should_continue(state: GraphState) -> str:
        if state["best_accuracy"] >= state["target_accuracy"]:
            print("\nЦелевая точность достигнута. Завершение.")
            return "end"
        if state["iteration"] > state["max_iterations"]:
            print("\nДостигнут лимит итераций. Завершение.")
            return "end"
        return "continue"

    workflow = StateGraph(GraphState)
    workflow.add_node("generate", generate_prompt_node)
    workflow.add_node("evaluate", evaluate_node)

    workflow.add_edge("generate", "evaluate")
    workflow.add_conditional_edges("evaluate", should_continue, {"continue": "generate", "end": END})
    workflow.set_entry_point("generate")

    return workflow.compile()