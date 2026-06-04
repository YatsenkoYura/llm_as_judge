import streamlit as st
import pandas as pd
import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from dataset import process_dataset
from graph import build_workflow
from models import GraphState

load_dotenv()
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="LLM Judge Optimizer", layout="wide")

# css
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
div[data-testid="stMetric"] {
    background: rgba(30, 30, 50, 0.6);
    border: 1px solid rgba(100, 100, 140, 0.25);
    border-radius: 10px;
    padding: 14px 18px;
}
.stTabs [data-baseweb="tab"] { font-weight: 500; padding: 8px 20px; }
</style>
""", unsafe_allow_html=True)


def render_confusion_matrix(tn, fp, fn, tp):
    # Кастомная CSS-визуализация Confusion Matrix
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 15px 0;">
        <div style="background: rgba(38, 166, 154, 0.12); border: 1px solid rgba(38, 166, 154, 0.3); border-radius: 8px; padding: 14px; text-align: center;">
            <div style="font-size: 0.85rem; font-weight: 500; color: #80cbc4;">True Negatives (TN)</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #26a69a; margin: 4px 0;">{tn}</div>
            <div style="font-size: 0.75rem; color: #b2dfdb;">Факт: 0 | Судья: 0</div>
        </div>
        <div style="background: rgba(239, 83, 80, 0.12); border: 1px solid rgba(239, 83, 80, 0.3); border-radius: 8px; padding: 14px; text-align: center;">
            <div style="font-size: 0.85rem; font-weight: 500; color: #ef9a9a;">False Positives (FP)</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #ef5350; margin: 4px 0;">{fp}</div>
            <div style="font-size: 0.75rem; color: #ffcdd2;">Факт: 0 | Судья: 1 (Ложная тревога)</div>
        </div>
        <div style="background: rgba(255, 167, 38, 0.12); border: 1px solid rgba(255, 167, 38, 0.3); border-radius: 8px; padding: 14px; text-align: center;">
            <div style="font-size: 0.85rem; font-weight: 500; color: #ffcc80;">False Negatives (FN)</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #ffa726; margin: 4px 0;">{fn}</div>
            <div style="font-size: 0.75rem; color: #ffe0b2;">Факт: 1 | Судья: 0 (Пропуск нарушения)</div>
        </div>
        <div style="background: rgba(38, 166, 154, 0.12); border: 1px solid rgba(38, 166, 154, 0.3); border-radius: 8px; padding: 14px; text-align: center;">
            <div style="font-size: 0.85rem; font-weight: 500; color: #80cbc4;">True Positives (TP)</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #26a69a; margin: 4px 0;">{tp}</div>
            <div style="font-size: 0.75rem; color: #b2dfdb;">Факт: 1 | Судья: 1</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# состояния
if "run_history" not in st.session_state:
    st.session_state.run_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "pipeline_logs" not in st.session_state:
    st.session_state.pipeline_logs = []

st.title("LLM Judge Optimizer")
st.caption("Итеративный поиск лучшего системного промпта для LLM-судьи")

tab_config, tab_data, tab_run, tab_results, tab_history = st.tabs([
    "Конфигурация", "Данные", "Запуск", "Результаты", "История"
])

# конфигурация
with tab_config:
    st.subheader("API")
    c1, c2 = st.columns(2)
    with c1:
        api_token = st.text_input("API Token", value=os.getenv("API_TOKEN", ""), type="password",
                                  help="Из .env или вручную")
        evaluator_model = st.text_input("Evaluator Model", value=os.getenv("EVALUATOR_MODEL", "gemini-3-flash-preview"))
    with c2:
        api_base_url = st.text_input("API Base URL", value=os.getenv("API_BASE_URL", "https://api.openai.com/v1"),
                                     help="Из .env")
        judge_model = st.text_input("Judge Model", value=os.getenv("JUDGE_MODEL", "gemini-3-flash-preview"))

    st.divider()
    st.subheader("Параметры пайплайна")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        num_train = st.number_input("Train выборка", min_value=1, value=10)
    with p2:
        num_val = st.number_input("Val выборка", min_value=1, value=20)
    with p3:
        max_iter = st.number_input("Макс. итераций", min_value=1, value=3)
    with p4:
        target_acc = st.number_input("Целевая точность, %", min_value=1.0, max_value=100.0, value=90.0)

    default_task = "Оцени диалог. 1 - агент решил задачу, 0 - не решил."
    task_desc = st.text_area("Описание задачи", value=default_task, height=80)

    st.divider()
    st.subheader("Продвинутые настройки")
    adv1, adv2, adv3 = st.columns(3)
    with adv1:
        use_shuffle = st.toggle("Перемешивать данные", value=True)
        shuffle_seed = st.number_input("Seed", min_value=0, value=42) if use_shuffle else 42
    with adv2:
        use_smart_truncation = st.toggle("Умная обрезка", value=True)
        truncation_max_len = st.number_input("Макс. длина (симв.)", min_value=200, value=1000, step=100) if use_smart_truncation else 1000
    with adv3:
        use_balanced_accuracy = st.toggle("Balanced Accuracy", value=True)
        max_parallel = st.number_input("Параллельных запросов", min_value=1, max_value=20, value=5)
        max_retries = st.number_input("Retry при ошибках", min_value=1, max_value=10, value=3)

    if api_token and api_token != "your_token_here":
        st.success("API Token настроен")
    else:
        st.warning("Введите API Token или задайте в .env")

# данные
with tab_data:
    st.subheader("Загрузка датасета")
    uploaded_file = st.file_uploader("CSV или JSON", type=["csv", "json"], label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_json(uploaded_file)
            st.session_state["uploaded_df"] = df
            st.session_state["uploaded_filename"] = uploaded_file.name

            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Строк", len(df))
            with c2: st.metric("Столбцов", len(df.columns))
            with c3: st.metric("Файл", uploaded_file.name)

            with st.expander("Превью", expanded=True):
                st.dataframe(df.head(5).astype(str), width="stretch")

            st.divider()
            st.subheader("Маппинг столбцов")
            columns = df.columns.tolist()

            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                context_col = st.selectbox("Столбец контекста", options=columns)
                drop_duplicates = st.checkbox("Удалить дубликаты", value=True, help="По контексту")
            with mc2:
                st.markdown("**Столбцы меток**")
                label_mode = st.radio("Режим:", ["Один столбец", "Несколько (any -> 1)"], label_visibility="collapsed")
                if label_mode == "Один столбец":
                    label_cols = [st.selectbox("Столбец оценки", options=columns)]
                else:
                    label_cols = st.multiselect("Столбцы оценки", options=columns)
            with mc3:
                explanation_col = st.selectbox("Столбец объяснения", options=columns)

            st.divider()
            st.subheader("Маппинг значений")
            mv1, mv2, mv3 = st.columns(3)
            with mv1:
                val_for_1_str = st.text_input("Значения для метки 1", value="Yes, 1, True")
                val_for_1 = [v.strip() for v in val_for_1_str.split(',')]
            with mv2:
                val_for_0_str = st.text_input("Значения для метки 0", value="No, 0, False")
                val_for_0 = [v.strip() for v in val_for_0_str.split(',')]
            with mv3:
                case_sensitive = st.checkbox("Регистрозависимый", value=False)

            st.session_state["col_mapping"] = {
                "context_col": context_col, "label_cols": label_cols,
                "explanation_col": explanation_col, "drop_duplicates": drop_duplicates,
                "val_for_1": val_for_1, "val_for_0": val_for_0, "case_sensitive": case_sensitive,
            }
            st.success("Данные загружены. Перейдите на вкладку «Запуск».")

        except Exception as e:
            st.error(f"Ошибка: {e}")
    else:
        st.info("Загрузите CSV или JSON для начала.")

# запуск
with tab_run:
    st.subheader("Запуск пайплайна")

    has_data = "uploaded_df" in st.session_state and "col_mapping" in st.session_state
    has_token = api_token and api_token != "your_token_here"

    if not has_data:
        st.warning("Сначала загрузите данные.")
    if not has_token:
        st.warning("Настройте API Token.")

    if has_data and has_token:
        mapping = st.session_state["col_mapping"]
        df_run = st.session_state["uploaded_df"].copy()

        with st.expander("Параметры запуска", expanded=False):
            pc1, pc2 = st.columns(2)
            with pc1:
                st.markdown(f"- **Evaluator:** `{evaluator_model}`\n- **Judge:** `{judge_model}`\n- **Train/Val:** {num_train}/{num_val}\n- **Итерации:** {max_iter}")
            with pc2:
                st.markdown(f"- **Целевая точность:** {target_acc}%\n- **Параллельность:** {max_parallel}\n- **Retry:** {max_retries}\n- **Balanced Accuracy:** {'да' if use_balanced_accuracy else 'нет'}")

        if st.button("Запустить оптимизацию", type="primary"):
            if mapping["drop_duplicates"]:
                df_run = df_run.drop_duplicates(subset=[mapping["context_col"]])
            if not mapping["label_cols"]:
                st.error("Выберите хотя бы один столбец оценки.")
                st.stop()

            train_data, val_data = process_dataset(
                df=df_run, context_col=mapping["context_col"],
                label_cols=mapping["label_cols"], explanation_col=mapping["explanation_col"],
                num_train=num_train, num_val=num_val,
                val_for_1=mapping["val_for_1"], val_for_0=mapping["val_for_0"],
                case_sensitive=mapping["case_sensitive"],
                shuffle=use_shuffle, shuffle_seed=shuffle_seed
            )

            if not train_data or not val_data:
                st.error("Выборка пуста.")
                st.stop()

            # статистика разбиения
            train_ones = sum(1 for ex in train_data if ex.expected_label == '1')
            val_ones = sum(1 for ex in val_data if ex.expected_label == '1')
            sc1, sc2 = st.columns(2)
            with sc1: st.info(f"Train: {len(train_data)} (0:{len(train_data)-train_ones} 1:{train_ones})")
            with sc2: st.info(f"Val: {len(val_data)} (0:{len(val_data)-val_ones} 1:{val_ones})")

            # прогресс
            status_container = st.status("Пайплайн запущен...", expanded=True)
            progress_bar = st.progress(0, text="Инициализация...")
            log_area = st.empty()
            logs = []

            def on_progress(event):
                etype = event.get("type", "")
                if etype == "generate_start":
                    msg = f"[iter {event['iteration']}] генерация промпта..."
                    status_container.update(label=msg)
                    logs.append(msg)
                elif etype == "generate_done":
                    it = event["iteration"]
                    msg = f"[iter {it}] промпт готов"
                    logs.append(msg)
                    progress_bar.progress(min(((it - 1) * 2 + 1) / (max_iter * 2), 1.0), text=msg)
                elif etype == "eval_start":
                    msg = f"[iter {event['iteration']}] оценка {event['total']} примеров..."
                    status_container.update(label=msg)
                    logs.append(msg)
                elif etype == "eval_progress":
                    cur, tot, it = event["current"], event["total"], event["iteration"]
                    if event.get("status") == "ok":
                        mark = "ok" if event.get("correct") else "miss"
                        logs.append(f"  [{cur}/{tot}] {mark}: exp={event['expected']} got={event['predicted']}")
                    else:
                        logs.append(f"  [{cur}/{tot}] api error")
                    base = ((it - 1) * 2 + 1) / (max_iter * 2)
                    step = (1 / (max_iter * 2)) * (cur / tot)
                    progress_bar.progress(min(base + step, 1.0), text=f"{cur}/{tot}")
                elif etype == "eval_done":
                    m = event.get("metrics", {})
                    if m:
                        logs.append(f"[iter {m['iteration']}] acc={m['accuracy']}% bacc={m['balanced_accuracy']}% f1={m['f1']}")
                elif etype == "pipeline_done":
                    logs.append(f"--- done: {event.get('reason')} ---")
                log_area.code("\n".join(logs[-25:]), language="log")

            # запуск графа
            client = OpenAI(api_key=api_token, base_url=api_base_url if api_base_url else None)
            initial_state = GraphState(
                task_description=task_desc,
                train_examples=train_data, val_examples=val_data,
                current_prompt="", feedback="",
                iteration=1, max_iterations=max_iter,
                target_accuracy=target_acc,
                best_prompt="", best_accuracy=0.0,
                metrics_history=[]
            )
            graph = build_workflow(
                client, evaluator_model, judge_model,
                use_smart_truncation=use_smart_truncation,
                truncation_max_len=truncation_max_len,
                use_balanced_accuracy=use_balanced_accuracy,
                max_retries=max_retries,
                max_parallel_workers=max_parallel,
                on_progress=on_progress
            )
            final_state = graph.invoke(initial_state)

            progress_bar.progress(1.0, text="Завершено")
            status_container.update(label="Пайплайн завершён", state="complete")

            # сохранение результата
            run_record = {
                "final_state": dict(final_state),
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "evaluator": evaluator_model, "judge": judge_model,
                    "train": num_train, "val": num_val, "iterations": max_iter,
                    "target": target_acc, "balanced": use_balanced_accuracy,
                },
                "logs": logs
            }
            st.session_state.last_result = run_record
            st.session_state.run_history.append(run_record)
            st.session_state.pipeline_logs = logs

            st.success("Готово. Перейдите на вкладку «Результаты».")

        st.divider()

        # бейзлайн
        if st.button("Тест на бейзлайне (zero-shot)"):
            if mapping["drop_duplicates"]:
                df_run = df_run.drop_duplicates(subset=[mapping["context_col"]])

            _, val_data = process_dataset(
                df=df_run, context_col=mapping["context_col"],
                label_cols=mapping["label_cols"], explanation_col=mapping["explanation_col"],
                num_train=num_train, num_val=num_val,
                val_for_1=mapping["val_for_1"], val_for_0=mapping["val_for_0"],
                case_sensitive=mapping["case_sensitive"],
                shuffle=use_shuffle, shuffle_seed=shuffle_seed
            )
            if not val_data:
                st.error("Валидационная выборка пуста.")
                st.stop()

            naive_prompt = (
                "Ты ИИ-судья. Оцени ответ агента в диалоге.\n"
                "Если агент справился с задачей — label 1, если нет — label 0.\n"
                "Ответь в JSON: {\"analysis\": \"...\", \"evidence\": \"...\", \"label\": \"0\" или \"1\"}"
            )

            from graph import _make_judge_caller
            client = OpenAI(api_key=api_token, base_url=api_base_url if api_base_url else None)
            call_judge = _make_judge_caller(client, judge_model, max_retries)

            tp = fp = tn = fn = api_errors = 0
            baseline_status = st.status("Бейзлайн: оценка...", expanded=True)
            baseline_progress = st.progress(0)

            from concurrent.futures import ThreadPoolExecutor, as_completed

            def eval_one(idx, ex):
                try:
                    r = call_judge(naive_prompt, ex.context)
                    return idx, ex, r, None
                except Exception as e:
                    return idx, ex, None, str(e)

            workers = min(max_parallel, len(val_data))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(eval_one, i, ex): i for i, ex in enumerate(val_data)}
                done = 0
                for future in as_completed(futures):
                    idx, example, result, error = future.result()
                    done += 1
                    baseline_progress.progress(done / len(val_data), text=f"{done}/{len(val_data)}")
                    if error:
                        api_errors += 1
                        continue
                    expected = float(example.expected_label)
                    predicted = float(result.label)
                    if expected == 1 and predicted == 1: tp += 1
                    elif expected == 0 and predicted == 1: fp += 1
                    elif expected == 0 and predicted == 0: tn += 1
                    elif expected == 1 and predicted == 0: fn += 1

            evaluated = len(val_data) - api_errors
            if evaluated == 0:
                st.error("Все запросы бейзлайна завершились ошибкой.")
            else:
                acc = ((tp + tn) / evaluated) * 100
                r1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                r0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                bacc = ((r0 + r1) / 2) * 100
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = r1
                f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

                baseline_status.update(label=f"Бейзлайн: acc={acc:.1f}% bacc={bacc:.1f}%", state="complete")

                st.subheader("Результаты бейзлайна (zero-shot)")
                bc1, bc2, bc3, bc4 = st.columns(4)
                with bc1: st.metric("Accuracy", f"{acc:.1f}%")
                with bc2: st.metric("Balanced Acc", f"{bacc:.1f}%")
                with bc3: st.metric("F1", f"{f1:.3f}")
                with bc4: st.metric("API Err", api_errors)

                render_confusion_matrix(tn, fp, fn, tp)
                
                with st.expander("Показать в табличном виде"):
                    cm_df = pd.DataFrame(
                        [[tn, fp], [fn, tp]],
                        index=['Факт: 0', 'Факт: 1'],
                        columns=['Предсказано: 0', 'Предсказано: 1']
                    )
                    st.dataframe(cm_df, width="stretch")

                # сравнение если есть результат пайплайна
                if st.session_state.last_result:
                    pipe_acc = st.session_state.last_result["final_state"]["best_accuracy"]
                    delta = pipe_acc - (bacc if use_balanced_accuracy else acc)
                    st.metric(
                        "Прирост пайплайна vs бейзлайн",
                        f"{pipe_acc:.1f}%",
                        delta=f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
                    )

                st.session_state["baseline_result"] = {
                    "accuracy": round(acc, 1), "balanced_accuracy": round(bacc, 1),
                    "f1": round(f1, 3), "precision": round(prec, 3), "recall": round(rec, 3),
                    "tp": tp, "fp": fp, "tn": tn, "fn": fn, "api_errors": api_errors
                }

# результаты
with tab_results:
    st.subheader("Результаты")

    if st.session_state.last_result is None:
        st.info("Запустите пайплайн для просмотра результатов.")
    else:
        res = st.session_state.last_result
        fs = res["final_state"]
        cfg = res["config"]

        metric_label = "Balanced Accuracy" if cfg["balanced"] else "Accuracy"
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric(metric_label, f"{fs['best_accuracy']:.1f}%")
        with m2: st.metric("Итераций", fs.get("iteration", 1) - 1)
        with m3: st.metric("Evaluator", cfg["evaluator"])
        with m4: st.metric("Judge", cfg["judge"])

        metrics_history = fs.get('metrics_history', [])
        if metrics_history:
            st.divider()
            st.subheader("Метрики по итерациям")
            metrics_df = pd.DataFrame(metrics_history)
            numeric_cols = ['iteration', 'accuracy', 'balanced_accuracy', 'precision',
                            'recall', 'f1', 'tp', 'fp', 'tn', 'fn', 'api_errors']
            display_df = metrics_df[[c for c in numeric_cols if c in metrics_df.columns]].copy()
            display_df.columns = ['Итерация', 'Acc %', 'BAcc %', 'Prec', 'Rec', 'F1',
                                  'TP', 'FP', 'TN', 'FN', 'Err'][:len(display_df.columns)]
            st.dataframe(display_df, width="stretch", hide_index=True)

            # график
            chart_df = metrics_df[['iteration', 'accuracy', 'balanced_accuracy', 'f1']].copy()
            chart_df.columns = ['iter', 'accuracy', 'bal_accuracy', 'f1']
            chart_df = chart_df.set_index('iter')
            chart_df['f1'] = chart_df['f1'] * 100
            st.line_chart(chart_df)

            # confusion matrix
            last = metrics_history[-1]
            st.subheader("Confusion Matrix (последняя итерация)")
            render_confusion_matrix(last['tn'], last['fp'], last['fn'], last['tp'])
            
            with st.expander("Показать в табличном виде"):
                cm_df = pd.DataFrame(
                    [[last['tn'], last['fp']], [last['fn'], last['tp']]],
                    index=['Факт: 0', 'Факт: 1'],
                    columns=['Предсказано: 0', 'Предсказано: 1']
                )
                st.dataframe(cm_df, width="stretch")

            # промпты
            st.divider()
            st.subheader("Промпты по итерациям")
            for m in metrics_history:
                snap = m.get("prompt_snapshot", "")
                if snap:
                    with st.expander(f"iter {m['iteration']} — acc:{m['accuracy']}% bacc:{m['balanced_accuracy']}%"):
                        st.code(snap, language="markdown")

        st.divider()
        st.subheader("Лучший промпт")
        st.code(fs['best_prompt'], language="markdown")

        # экспорт
        st.divider()
        st.subheader("Экспорт")
        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            st.download_button("Промпт (.txt)", data=fs['best_prompt'],
                               file_name="best_prompt.txt", mime="text/plain")
        with dl2:
            if metrics_history:
                export_df = pd.DataFrame(metrics_history).drop(columns=['prompt_snapshot'], errors='ignore')
                st.download_button("Метрики (.csv)", data=export_df.to_csv(index=False),
                                   file_name="metrics.csv", mime="text/csv")
        with dl3:
            report = {"best_prompt": fs['best_prompt'], "best_accuracy": fs['best_accuracy'],
                      "metrics_history": metrics_history, "config": cfg, "timestamp": res["timestamp"]}
            st.download_button("Отчёт (.json)", data=json.dumps(report, ensure_ascii=False, indent=2),
                               file_name="report.json", mime="application/json")

        with open("best_prompt.txt", "w", encoding="utf-8") as f:
            f.write(fs['best_prompt'])

# история
with tab_history:
    st.subheader("История запусков")

    if not st.session_state.run_history:
        st.info("История пуста.")
    else:
        for i, run in enumerate(reversed(st.session_state.run_history)):
            fs_h = run["final_state"]
            cfg_h = run["config"]
            ts = run.get("timestamp", "—")
            with st.expander(f"{ts} | acc:{fs_h['best_accuracy']:.1f}% | {cfg_h['evaluator']}/{cfg_h['judge']}", expanded=(i == 0)):
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Best Accuracy", f"{fs_h['best_accuracy']:.1f}%")
                with c2: st.metric("Итераций", fs_h.get('iteration', 1) - 1)
                with c3: st.metric("Train/Val", f"{cfg_h['train']}/{cfg_h['val']}")

                hist_metrics = fs_h.get('metrics_history', [])
                if hist_metrics:
                    hist_df = pd.DataFrame(hist_metrics).drop(columns=['prompt_snapshot'], errors='ignore')
                    st.dataframe(hist_df, width="stretch", hide_index=True)

                run_logs = run.get("logs", [])
                if run_logs:
                    with st.expander("Логи"):
                        st.code("\n".join(run_logs), language="log")
