import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional, Dict, Any

from langgraph.graph import StateGraph, END
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from models import GraphState, JudgeResponse

logger = logging.getLogger(__name__)


def _smart_truncate(text: str, max_len: int = 1000) -> str:
    """берёт начало + конец текста, чтобы не терять контекст."""
    if len(text) <= max_len:
        return text
    half = max_len // 2
    return text[:half] + "\n...[обрезано]...\n" + text[-half:]


def _make_judge_caller(client: OpenAI, judge_model: str, max_retries: int = 3):
    """обёртка вызова судьи с retry."""

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
    call_judge = _make_judge_caller(client, judge_model, max_retries)

    def _emit(event: dict):
        if on_progress:
            try:
                on_progress(event)
            except Exception:
                pass

    # генерация промпта
    def generate_prompt_node(state: GraphState) -> dict:
        iteration = state['iteration']
        logger.info(f"[iter {iteration}] generating prompt...")
        _emit({"type": "generate_start", "iteration": iteration})

        examples_blocks = []
        for ex in state['train_examples']:
            ctx = _smart_truncate(ex.context, truncation_max_len) if use_smart_truncation else ex.context[:700]
            block = f"Диалог/Логи: {ctx}\nОжидаемая оценка: {ex.expected_label}\nЛогика разметки: {ex.explanation}"
            examples_blocks.append(block)

        examples_text = "\n\n".join(examples_blocks)

        system_message = f"""
        Ты эксперт по машинному обучению. Напиши системную инструкцию для LLM-судьи.
        Задача, которую решает оцениваемый агент: {state['task_description']}

        ВАЖНО: Судья будет отвечать строго в JSON-формате с полями:
        - "analysis": краткий тезисный анализ (буллиты)
        - "evidence": ключевые цитаты/факты из текста, подтверждающие вывод
        - "label": "0" или "1"
        Твоя инструкция должна направлять судью заполнять именно эти поля. НЕ описывай другой формат ответа.

        ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА ДЛЯ ТВОЕЙ ИНСТРУКЦИИ:
        1. Четко следуй 'Описанию задачи', переданному выше.
        2. Краткость: запрети судье лить воду. Рассуждения тезисные (буллиты).
        3. Few-Shot: включи 2 коротких выдуманных примера оценки (на 0 и на 1) в JSON-формате.

        Исторические данные разметки:
        {examples_text}
        """

        if state['iteration'] > 1 and state['current_prompt']:
            system_message += f"""
        Предыдущая версия инструкции:
        ---
        {state['current_prompt']}
        ---

        Ошибки на валидации:
        {state['feedback']}

        Задача: улучши предыдущую инструкцию, добавив правила для исправления ошибок.
        Сохрани основной каркас — он уже работает на многих примерах.
        Выведи только текст новой инструкции.
        """
        else:
            system_message += "\nВыведи только текст инструкции."

        response = client.chat.completions.create(
            model=evaluator_model,
            messages=[{"role": "user", "content": system_message}],
            temperature=0.4
        )
        generated_prompt = response.choices[0].message.content.strip()
        logger.info(f"prompt generated, length={len(generated_prompt)}")

        _emit({
            "type": "generate_done", "iteration": iteration,
            "prompt_preview": generated_prompt[:200] + "..." if len(generated_prompt) > 200 else generated_prompt
        })
        return {"current_prompt": generated_prompt}

    # оценка на валидации
    def evaluate_node(state: GraphState) -> dict:
        iteration = state['iteration']
        logger.info(f"[iter {iteration}] evaluating...")
        _emit({"type": "eval_start", "iteration": iteration, "total": len(state['val_examples'])})

        tp = fp = tn = fn = api_errors = 0
        errors = []
        total = len(state['val_examples'])

        def eval_single(idx, example):
            try:
                result = call_judge(state['current_prompt'], example.context)
                return idx, example, result, None
            except Exception as e:
                return idx, example, None, str(e)

        # параллельная оценка
        workers = min(max_parallel_workers, total)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(eval_single, i, ex): i for i, ex in enumerate(state['val_examples'])}
            completed = 0
            for future in as_completed(futures):
                idx, example, result, error = future.result()
                completed += 1

                if error:
                    api_errors += 1
                    logger.warning(f"fail after {max_retries} retries: {error}")
                    _emit({"type": "eval_progress", "iteration": iteration,
                           "current": completed, "total": total, "status": "error", "error": error})
                    continue

                expected = float(example.expected_label)
                predicted = float(result.label)
                is_correct = predicted == expected

                # confusion matrix
                if expected == 1 and predicted == 1: tp += 1
                elif expected == 0 and predicted == 1: fp += 1
                elif expected == 0 and predicted == 0: tn += 1
                elif expected == 1 and predicted == 0: fn += 1

                logger.info(f"exp={example.expected_label} pred={result.label} {'ok' if is_correct else 'miss'}")

                if not is_correct:
                    errors.append(
                        f"Ожидали: {example.expected_label}, Получили: {result.label}\n"
                        f"Анализ: {result.analysis}\nДоказательства: {result.evidence}"
                    )

                _emit({"type": "eval_progress", "iteration": iteration,
                       "current": completed, "total": total, "status": "ok",
                       "expected": example.expected_label, "predicted": result.label,
                       "correct": is_correct, "analysis_preview": result.analysis[:100]})

        # метрики
        evaluated = total - api_errors
        if evaluated == 0:
            logger.warning("all requests failed")
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

        primary_metric = balanced_accuracy if use_balanced_accuracy else accuracy
        metric_name = "bal_acc" if use_balanced_accuracy else "acc"

        logger.info(f"[iter {iteration}] {metric_name}={primary_metric:.1f}% acc={accuracy:.1f}% "
                     f"prec={precision:.3f} rec={recall:.3f} f1={f1:.3f} "
                     f"tp={tp} fp={fp} tn={tn} fn={fn} errors={api_errors}")

        error_stats = (
            f"FP (ожидали 0, получили 1): {fp}, FN (ожидали 1, получили 0): {fn}\n"
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

    # условие остановки
    def should_continue(state: GraphState) -> str:
        if state["best_accuracy"] >= state["target_accuracy"]:
            logger.info("target reached")
            _emit({"type": "pipeline_done", "reason": "target_reached"})
            return "end"
        if state["iteration"] > state["max_iterations"]:
            logger.info("max iterations reached")
            _emit({"type": "pipeline_done", "reason": "max_iterations"})
            return "end"
        return "continue"

    # граф
    workflow = StateGraph(GraphState)
    workflow.add_node("generate", generate_prompt_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_edge("generate", "evaluate")
    workflow.add_conditional_edges("evaluate", should_continue, {"continue": "generate", "end": END})
    workflow.set_entry_point("generate")
    return workflow.compile()