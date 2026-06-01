import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional, Dict, Any

from langgraph.graph import StateGraph, END
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from models import GraphState, JudgeResponse

logger = logging.getLogger(__name__)


def _smart_truncate(text: str, max_len: int = 1000) -> str:
    """Умная обрезка: берёт начало и конец текста, чтобы не терять важный контекст."""
    if len(text) <= max_len:
        return text
    half = max_len // 2
    return text[:half] + "\n...[обрезано]...\n" + text[-half:]


def _make_judge_caller(client: OpenAI, judge_model: str, max_retries: int = 3):
    """Factory: creates a retry-enabled judge call function."""

    @retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def call_judge(prompt: str, context: str) -> JudgeResponse:
        response = client.beta.chat.completions.parse(
            model=judge_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Оцени этот лог/диалог:\n{context}"}
            ],
            response_format=JudgeResponse,
            temperature=0.0
        )
        return response.choices[0].message.parsed

    return call_judge


def build_workflow(
    client: OpenAI,
    evaluator_model: str,
    judge_model: str,
    use_smart_truncation: bool = True,
    truncation_max_len: int = 1000,
    use_balanced_accuracy: bool = True,
    max_retries: int = 3,
    max_parallel_workers: int = 5,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None
):
    """
    Build the LangGraph workflow with retry logic, parallel eval, and progress callbacks.

    on_progress: Optional callback receiving dicts like:
        {"type": "generate_start", "iteration": 1}
        {"type": "generate_done", "iteration": 1, "prompt_preview": "..."}
        {"type": "eval_progress", "iteration": 1, "current": 5, "total": 20, ...}
        {"type": "eval_done", "iteration": 1, "metrics": {...}}
    """
    call_judge = _make_judge_caller(client, judge_model, max_retries)

    def _emit(event: dict):
        if on_progress:
            try:
                on_progress(event)
            except Exception:
                pass  # Never break the pipeline because of a UI callback

    def generate_prompt_node(state: GraphState) -> dict:
        iteration = state['iteration']
        logger.info(f"[Итерация {iteration}] Генерация инструкции...")
        _emit({"type": "generate_start", "iteration": iteration})

        examples_blocks = []
        for ex in state['train_examples']:
            context_display = _smart_truncate(ex.context, truncation_max_len) if use_smart_truncation else ex.context[:700]
            block = f"""
            Диалог/Логи: {context_display}
            Ожидаемая оценка: {ex.expected_label}
            Логика разметки: {ex.explanation}
            """
            examples_blocks.append(block.strip())

        examples_text = "\n\n".join(examples_blocks)

        response_format_description = """
        ВАЖНО: Судья будет отвечать строго в JSON-формате с полями:
        - "analysis": краткий тезисный анализ (буллиты)
        - "evidence": ключевые цитаты/факты из текста, подтверждающие вывод
        - "label": "0" или "1"
        Твоя инструкция должна направлять судью заполнять именно эти поля. НЕ описывай другой формат ответа.
        """

        system_message = f"""
        Ты эксперт по машинному обучению. Напиши системную инструкцию для LLM-судьи.
        Задача, которую решает оцениваемый агент: {state['task_description']}

        {response_format_description}

        ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА ДЛЯ ТВОЕЙ ИНСТРУКЦИИ:
        1. Четко следуй 'Описанию задачи', переданному выше. Инструкция должна быть сфокусирована именно на этой задаче.
        2. Краткость: Запрети судье лить воду. Рассуждения должны быть тезисными (буллиты).
        3. Few-Shot примеры: ОБЯЗАТЕЛЬНО включи прямо внутрь своей инструкции 2 коротких выдуманных примера оценки (один на 0, другой на 1), чтобы показать судье шаблон рассуждений. Примеры должны использовать JSON-формат с полями analysis, evidence, label.

        Исторические данные разметки для понимания задачи:
        {examples_text}
        """

        if state['iteration'] > 1 and state['current_prompt']:
            system_message += f"""
        Твоя ПРЕДЫДУЩАЯ версия инструкции:
        ---
        {state['current_prompt']}
        ---

        Она дала ошибки на валидации. Вот статистика ошибок:
        {state['feedback']}

        Твоя задача: УЛУЧШИТЬ предыдущую инструкцию. 
        Добавь в нее новые правила, чтобы исправить эти ошибки, НО обязательно сохрани основной каркас, так как на многих примерах он уже работает правильно. Не начинай с нуля!
        Выведи только текст новой инструкции.
        """
        else:
            system_message += "\nВыведи только текст инструкции (без лишних приветствий)."

        response = client.chat.completions.create(
            model=evaluator_model,
            messages=[{"role": "user", "content": system_message}],
            temperature=0.4
        )

        generated_prompt = response.choices[0].message.content.strip()

        logger.info("СГЕНЕРИРОВАННЫЙ ПРОМПТ:")
        logger.info(generated_prompt if generated_prompt else "[ВНИМАНИЕ: ИИ ВЕРНУЛ ПУСТУЮ СТРОКУ!]")

        _emit({
            "type": "generate_done",
            "iteration": iteration,
            "prompt_preview": generated_prompt[:200] + "..." if len(generated_prompt) > 200 else generated_prompt
        })

        return {"current_prompt": generated_prompt}

    def evaluate_node(state: GraphState) -> dict:
        iteration = state['iteration']
        logger.info(f"[Итерация {iteration}] Оценка на валидационной выборке...")
        _emit({"type": "eval_start", "iteration": iteration, "total": len(state['val_examples'])})

        tp = fp = tn = fn = api_errors = 0
        errors = []
        total = len(state['val_examples'])
        results_collected = []

        # --- Parallel evaluation ---
        def eval_single(idx, example):
            try:
                result = call_judge(state['current_prompt'], example.context)
                return idx, example, result, None
            except Exception as e:
                return idx, example, None, str(e)

        workers = min(max_parallel_workers, total)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(eval_single, i, ex): i
                for i, ex in enumerate(state['val_examples'])
            }
            completed = 0
            for future in as_completed(futures):
                idx, example, result, error = future.result()
                completed += 1

                if error:
                    api_errors += 1
                    logger.warning(f"❌ ОШИБКА (после {max_retries} попыток): {error}")
                    _emit({
                        "type": "eval_progress", "iteration": iteration,
                        "current": completed, "total": total,
                        "status": "error", "error": error
                    })
                    continue

                expected = float(example.expected_label)
                predicted = float(result.label)
                is_correct = predicted == expected

                if expected == 1 and predicted == 1:
                    tp += 1
                elif expected == 0 and predicted == 1:
                    fp += 1
                elif expected == 0 and predicted == 0:
                    tn += 1
                elif expected == 1 and predicted == 0:
                    fn += 1

                logger.info(f"Ожидали: {example.expected_label} | ИИ: {result.label} | {'✅' if is_correct else '❌'}")

                if not is_correct:
                    errors.append(
                        f"Ожидали: {example.expected_label}, Получили: {result.label}\n"
                        f"Анализ судьи: {result.analysis}\n"
                        f"Доказательства: {result.evidence}"
                    )

                _emit({
                    "type": "eval_progress", "iteration": iteration,
                    "current": completed, "total": total,
                    "status": "ok",
                    "expected": example.expected_label,
                    "predicted": result.label,
                    "correct": is_correct,
                    "analysis_preview": result.analysis[:100]
                })

        # --- Metrics ---
        evaluated = total - api_errors
        if evaluated == 0:
            logger.warning("⚠️ Все запросы завершились ошибкой!")
            _emit({"type": "eval_done", "iteration": iteration, "error": "all_failed"})
            return {"iteration": state["iteration"] + 1, "feedback": "Все запросы завершились ошибкой API."}

        correct = tp + tn
        accuracy = (correct / evaluated) * 100

        recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        balanced_accuracy = ((recall_0 + recall_1) / 2) * 100

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = recall_1
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        if use_balanced_accuracy:
            primary_metric = balanced_accuracy
            metric_name = "Balanced Accuracy"
        else:
            primary_metric = accuracy
            metric_name = "Accuracy"

        logger.info(f"\n--- Результаты итерации {iteration} ---")
        logger.info(f"Accuracy: {accuracy:.1f}% | Balanced Accuracy: {balanced_accuracy:.1f}% | ({metric_name}): {primary_metric:.1f}%")
        logger.info(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
        logger.info(f"TP={tp} FP={fp} TN={tn} FN={fn} | API ошибок: {api_errors}")

        error_stats = (
            f"Статистика ошибок: Ложноположительных (FP, ожидали 0, получили 1): {fp}, "
            f"Ложноотрицательных (FN, ожидали 1, получили 0): {fn}\n"
            f"Balanced Accuracy: {balanced_accuracy:.1f}% | F1: {f1:.3f}\n"
        )
        error_examples = "\n\n".join(errors[:3]) if errors else ""
        feedback = error_stats + "\nПримеры ошибок:\n" + error_examples if error_examples else error_stats

        iteration_metrics = {
            "iteration": iteration,
            "accuracy": round(accuracy, 1),
            "balanced_accuracy": round(balanced_accuracy, 1),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "api_errors": api_errors,
            "prompt_snapshot": state["current_prompt"]
        }
        history = list(state.get("metrics_history", []))
        history.append(iteration_metrics)

        updates = {
            "iteration": state["iteration"] + 1,
            "feedback": feedback,
            "metrics_history": history
        }

        if primary_metric > state["best_accuracy"]:
            updates["best_accuracy"] = primary_metric
            updates["best_prompt"] = state["current_prompt"]

        _emit({"type": "eval_done", "iteration": iteration, "metrics": iteration_metrics})

        return updates

    def should_continue(state: GraphState) -> str:
        if state["best_accuracy"] >= state["target_accuracy"]:
            logger.info("Целевая точность достигнута. Завершение.")
            _emit({"type": "pipeline_done", "reason": "target_reached"})
            return "end"
        if state["iteration"] > state["max_iterations"]:
            logger.info("Достигнут лимит итераций. Завершение.")
            _emit({"type": "pipeline_done", "reason": "max_iterations"})
            return "end"
        return "continue"

    workflow = StateGraph(GraphState)
    workflow.add_node("generate", generate_prompt_node)
    workflow.add_node("evaluate", evaluate_node)

    workflow.add_edge("generate", "evaluate")
    workflow.add_conditional_edges("evaluate", should_continue, {"continue": "generate", "end": END})
    workflow.set_entry_point("generate")

    return workflow.compile()