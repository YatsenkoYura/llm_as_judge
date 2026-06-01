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
from models import GraphState, RunConfig

load_dotenv()
logging.basicConfig(level=logging.INFO)

# ── Page Config ──
st.set_page_config(page_title="LLM Judge Optimizer", page_icon="⚖️", layout="wide")

# ── Custom CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}
div[data-testid="stMetric"] label { color: #a5b4fc !important; font-weight: 500; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #e0e7ff !important; font-weight: 700; }

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(15, 23, 42, 0.4);
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 500;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #7c3aed);
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.6rem 2rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
}
div.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 25px rgba(99, 102, 241, 0.5);
    transform: translateY(-1px);
}

div[data-testid="stExpander"] {
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 10px;
    overflow: hidden;
}

.success-banner {
    background: linear-gradient(135deg, #065f46, #047857);
    border: 1px solid #34d399;
    border-radius: 12px;
    padding: 16px 24px;
    color: #d1fae5;
    font-weight: 600;
    text-align: center;
    margin: 1rem 0;
}
.header-gradient {
    background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}
.subtitle { color: #94a3b8; font-size: 1.05rem; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Session State Init ──
if "run_history" not in st.session_state:
    st.session_state.run_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "pipeline_logs" not in st.session_state:
    st.session_state.pipeline_logs = []

# ── Header ──
st.markdown('<div class="header-gradient">⚖️ LLM Judge Optimizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Итеративный поиск лучшего системного промпта для LLM-судьи</div>', unsafe_allow_html=True)

# ── Tabs ──
tab_config, tab_data, tab_run, tab_results, tab_history = st.tabs([
    "⚙️ Конфигурация", "📂 Данные", "🚀 Запуск", "📊 Результаты", "📜 История"
])

# ══════════════════════════════════════════════
# TAB 1: Configuration
# ══════════════════════════════════════════════
with tab_config:
    st.subheader("🔑 API")
    c1, c2 = st.columns(2)
    with c1:
        api_token = st.text_input("API Token", value=os.getenv("API_TOKEN", ""), type="password",
                                  help="Токен из .env (API_TOKEN) или введите вручную")
        evaluator_model = st.text_input("Evaluator Model", value=os.getenv("EVALUATOR_MODEL", "gemini-3-flash-preview"))
    with c2:
        api_base_url = st.text_input("API Base URL", value=os.getenv("API_BASE_URL", "https://api.openai.com/v1"),
                                     help="Берётся из .env (API_BASE_URL)")
        judge_model = st.text_input("Judge Model", value=os.getenv("JUDGE_MODEL", "gemini-3-flash-preview"))

    st.divider()
    st.subheader("🎛️ Параметры пайплайна")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        num_train = st.number_input("Train выборка", min_value=1, value=10)
    with p2:
        num_val = st.number_input("Val выборка", min_value=1, value=20)
    with p3:
        max_iter = st.number_input("Макс. итераций", min_value=1, value=3)
    with p4:
        target_acc = st.number_input("Целевая точность %", min_value=1.0, max_value=100.0, value=90.0)

    default_task = "Оцени диалог. 1 - агент решил задачу, 0 - не решил."
    task_desc = st.text_area("📝 Описание задачи", value=default_task, height=80)

    st.divider()
    st.subheader("🔧 Продвинутые настройки")
    adv1, adv2, adv3 = st.columns(3)
    with adv1:
        use_shuffle = st.toggle("Перемешивать данные", value=True)
        if use_shuffle:
            shuffle_seed = st.number_input("Seed", min_value=0, value=42)
        else:
            shuffle_seed = 42
    with adv2:
        use_smart_truncation = st.toggle("Умная обрезка", value=True)
        if use_smart_truncation:
            truncation_max_len = st.number_input("Макс. длина (симв.)", min_value=200, value=1000, step=100)
        else:
            truncation_max_len = 1000
    with adv3:
        use_balanced_accuracy = st.toggle("Balanced Accuracy", value=True)
        max_parallel = st.number_input("Параллельных запросов", min_value=1, max_value=20, value=5)
        max_retries = st.number_input("Retry при ошибках", min_value=1, max_value=10, value=3)

    if api_token and api_token != "your_token_here":
        st.success("✅ API Token настроен")
    else:
        st.warning("⚠️ Введите API Token (или задайте в .env)")

# ══════════════════════════════════════════════
# TAB 2: Data
# ══════════════════════════════════════════════
with tab_data:
    st.subheader("📁 Загрузка датасета")
    uploaded_file = st.file_uploader("CSV или JSON файл", type=["csv", "json"], label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_json(uploaded_file)

            st.session_state["uploaded_df"] = df
            st.session_state["uploaded_filename"] = uploaded_file.name

            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("Строк", len(df))
            with col_info2:
                st.metric("Столбцов", len(df.columns))
            with col_info3:
                st.metric("Файл", uploaded_file.name)

            with st.expander("👀 Превью данных", expanded=True):
                st.dataframe(df.head(5), use_container_width=True)

            st.divider()
            st.subheader("🏷️ Маппинг столбцов")
            columns = df.columns.tolist()

            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                context_col = st.selectbox("Столбец контекста", options=columns)
                drop_duplicates = st.checkbox("Удалить дубликаты", value=True,
                                              help="Убирает дубли по контексту (DICES и т.п.)")
            with mc2:
                st.markdown("**Столбцы меток**")
                label_mode = st.radio("Режим:", ["Один столбец", "Несколько (any → 1)"], label_visibility="collapsed")
                if label_mode == "Один столбец":
                    label_cols = [st.selectbox("Столбец оценки", options=columns)]
                else:
                    label_cols = st.multiselect("Столбцы оценки", options=columns)
            with mc3:
                explanation_col = st.selectbox("Столбец объяснения", options=columns)

            st.divider()
            st.subheader("🔄 Маппинг значений")
            mv1, mv2, mv3 = st.columns(3)
            with mv1:
                val_for_1_str = st.text_input("Значения → 1", value="Yes, 1, True")
                val_for_1 = [v.strip() for v in val_for_1_str.split(',')]
            with mv2:
                val_for_0_str = st.text_input("Значения → 0", value="No, 0, False")
                val_for_0 = [v.strip() for v in val_for_0_str.split(',')]
            with mv3:
                case_sensitive = st.checkbox("Регистрозависимый", value=False)

            # Store mapping in session
            st.session_state["col_mapping"] = {
                "context_col": context_col, "label_cols": label_cols,
                "explanation_col": explanation_col, "drop_duplicates": drop_duplicates,
                "val_for_1": val_for_1, "val_for_0": val_for_0, "case_sensitive": case_sensitive,
            }
            st.success("✅ Данные загружены и маппинг настроен. Перейдите на вкладку **🚀 Запуск**")

        except Exception as e:
            st.error(f"Ошибка при обработке файла: {str(e)}")
    else:
        st.info("Загрузите CSV или JSON файл для начала работы.")

# ══════════════════════════════════════════════
# TAB 3: Run Pipeline
# ══════════════════════════════════════════════
with tab_run:
    st.subheader("🚀 Запуск пайплайна")

    # Pre-flight checks
    has_data = "uploaded_df" in st.session_state and "col_mapping" in st.session_state
    has_token = api_token and api_token != "your_token_here"

    if not has_data:
        st.warning("📂 Сначала загрузите данные на вкладке **📂 Данные**")
    if not has_token:
        st.warning("🔑 Настройте API Token на вкладке **⚙️ Конфигурация**")

    if has_data and has_token:
        mapping = st.session_state["col_mapping"]
        df_run = st.session_state["uploaded_df"].copy()

        with st.expander("📋 Текущие параметры запуска", expanded=False):
            pc1, pc2 = st.columns(2)
            with pc1:
                st.markdown(f"""
                - **Evaluator:** `{evaluator_model}`
                - **Judge:** `{judge_model}`
                - **Train/Val:** {num_train}/{num_val}
                - **Итерации:** {max_iter}
                """)
            with pc2:
                st.markdown(f"""
                - **Целевая точность:** {target_acc}%
                - **Параллельность:** {max_parallel} потоков
                - **Retry:** {max_retries} попыток
                - **Balanced Accuracy:** {'Да' if use_balanced_accuracy else 'Нет'}
                """)

        if st.button("🚀 Запустить оптимизацию", type="primary", use_container_width=True):
            # Prepare data
            if mapping["drop_duplicates"]:
                df_run = df_run.drop_duplicates(subset=[mapping["context_col"]])

            if not mapping["label_cols"]:
                st.error("Выберите хотя бы один столбец оценки.")
                st.stop()

            train_data, val_data = process_dataset(
                df=df_run,
                context_col=mapping["context_col"],
                label_cols=mapping["label_cols"],
                explanation_col=mapping["explanation_col"],
                num_train=num_train, num_val=num_val,
                val_for_1=mapping["val_for_1"], val_for_0=mapping["val_for_0"],
                case_sensitive=mapping["case_sensitive"],
                shuffle=use_shuffle, shuffle_seed=shuffle_seed
            )

            if len(train_data) == 0 or len(val_data) == 0:
                st.error("Обучающая или валидационная выборка пуста!")
                st.stop()

            # Stats
            train_ones = sum(1 for ex in train_data if ex.expected_label == '1')
            val_ones = sum(1 for ex in val_data if ex.expected_label == '1')
            sc1, sc2 = st.columns(2)
            with sc1:
                st.info(f"📊 Train: {len(train_data)} (0: {len(train_data)-train_ones}, 1: {train_ones})")
            with sc2:
                st.info(f"📊 Val: {len(val_data)} (0: {len(val_data)-val_ones}, 1: {val_ones})")

            # Progress UI
            status_container = st.status("🔄 Пайплайн запущен...", expanded=True)
            progress_bar = st.progress(0, text="Инициализация...")
            log_area = st.empty()
            logs = []

            def on_progress(event):
                etype = event.get("type", "")
                if etype == "generate_start":
                    it = event["iteration"]
                    msg = f"🧠 Итерация {it}: генерация промпта..."
                    status_container.update(label=msg)
                    logs.append(msg)
                elif etype == "generate_done":
                    it = event["iteration"]
                    msg = f"✅ Итерация {it}: промпт сгенерирован"
                    logs.append(msg)
                    pct = ((it - 1) * 2 + 1) / (max_iter * 2)
                    progress_bar.progress(min(pct, 1.0), text=msg)
                elif etype == "eval_start":
                    msg = f"📊 Итерация {event['iteration']}: оценка {event['total']} примеров..."
                    status_container.update(label=msg)
                    logs.append(msg)
                elif etype == "eval_progress":
                    cur, tot = event["current"], event["total"]
                    it = event["iteration"]
                    if event.get("status") == "ok":
                        icon = "✅" if event.get("correct") else "❌"
                        detail = f"  {icon} [{cur}/{tot}] ожидали={event['expected']} получили={event['predicted']}"
                    else:
                        detail = f"  ⚠️ [{cur}/{tot}] ошибка API"
                    logs.append(detail)
                    base = ((it - 1) * 2 + 1) / (max_iter * 2)
                    step = (1 / (max_iter * 2)) * (cur / tot)
                    progress_bar.progress(min(base + step, 1.0), text=f"Пример {cur}/{tot}")
                elif etype == "eval_done":
                    m = event.get("metrics", {})
                    if m:
                        msg = f"📈 Итерация {m.get('iteration')}: Acc={m.get('accuracy')}% BAcc={m.get('balanced_accuracy')}% F1={m.get('f1')}"
                        logs.append(msg)
                elif etype == "pipeline_done":
                    logs.append(f"🏁 Завершено: {event.get('reason')}")
                log_area.code("\n".join(logs[-20:]), language="log")

            # Build & Run
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

            progress_bar.progress(1.0, text="✅ Завершено!")
            status_container.update(label="✅ Пайплайн завершён!", state="complete")

            # Save to session
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
            # Remove non-serializable items for display
            display_state = {k: v for k, v in final_state.items() if k not in ['train_examples', 'val_examples']}
            run_record["display_state"] = display_state
            st.session_state.last_result = run_record
            st.session_state.run_history.append(run_record)
            st.session_state.pipeline_logs = logs

            st.markdown('<div class="success-banner">🎉 Пайплайн успешно завершён! Перейдите на вкладку 📊 Результаты</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 4: Results
# ══════════════════════════════════════════════
with tab_results:
    st.subheader("📊 Результаты")

    if st.session_state.last_result is None:
        st.info("Запустите пайплайн на вкладке **🚀 Запуск** для просмотра результатов.")
    else:
        res = st.session_state.last_result
        fs = res["final_state"]
        cfg = res["config"]

        # Key metrics
        metric_label = "Balanced Accuracy" if cfg["balanced"] else "Accuracy"
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(f"🏆 {metric_label}", f"{fs['best_accuracy']:.1f}%")
        with m2:
            total_iters = fs.get("iteration", 1) - 1
            st.metric("🔄 Итераций", total_iters)
        with m3:
            st.metric("🤖 Evaluator", cfg["evaluator"])
        with m4:
            st.metric("⚖️ Judge", cfg["judge"])

        metrics_history = fs.get('metrics_history', [])
        if metrics_history:
            st.divider()
            st.subheader("📈 Метрики по итерациям")
            metrics_df = pd.DataFrame(metrics_history)
            display_cols = ['iteration', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'tp', 'fp', 'tn', 'fn', 'api_errors']
            display_df = metrics_df[[c for c in display_cols if c in metrics_df.columns]].copy()
            display_df.columns = ['Итерация', 'Accuracy %', 'Bal.Acc %', 'Precision', 'Recall', 'F1', 'TP', 'FP', 'TN', 'FN', 'API Err'][:len(display_df.columns)]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Chart
            chart_df = metrics_df[['iteration', 'accuracy', 'balanced_accuracy', 'f1']].copy()
            chart_df = chart_df.rename(columns={'iteration': 'Итерация', 'accuracy': 'Accuracy', 'balanced_accuracy': 'Bal.Accuracy', 'f1': 'F1'})
            chart_df = chart_df.set_index('Итерация')
            chart_df['F1'] = chart_df['F1'] * 100
            st.line_chart(chart_df)

            # Confusion Matrix
            last = metrics_history[-1]
            st.subheader("🔢 Confusion Matrix (последняя итерация)")
            cm_df = pd.DataFrame(
                [[last['tn'], last['fp']], [last['fn'], last['tp']]],
                index=['Факт: 0', 'Факт: 1'],
                columns=['Предсказано: 0', 'Предсказано: 1']
            )
            st.dataframe(cm_df, use_container_width=True)

            # Prompt versions
            st.divider()
            st.subheader("📝 Промпты по итерациям")
            for m in metrics_history:
                snap = m.get("prompt_snapshot", "")
                if snap:
                    with st.expander(f"Итерация {m['iteration']} — Acc: {m['accuracy']}% | BAcc: {m['balanced_accuracy']}%"):
                        st.code(snap, language="markdown")

        # Best prompt
        st.divider()
        st.subheader("🏆 Лучший промпт")
        st.code(fs['best_prompt'], language="markdown")

        # Export buttons
        st.divider()
        st.subheader("📥 Экспорт")
        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            st.download_button("📥 Лучший промпт (.txt)", data=fs['best_prompt'],
                               file_name="best_prompt.txt", mime="text/plain", use_container_width=True)
        with dl2:
            if metrics_history:
                csv_data = pd.DataFrame(metrics_history).to_csv(index=False)
                st.download_button("📥 Метрики (.csv)", data=csv_data,
                                   file_name="metrics_history.csv", mime="text/csv", use_container_width=True)
        with dl3:
            report = {
                "best_prompt": fs['best_prompt'],
                "best_accuracy": fs['best_accuracy'],
                "metrics_history": metrics_history,
                "config": cfg, "timestamp": res["timestamp"]
            }
            st.download_button("📥 Полный отчёт (.json)",
                               data=json.dumps(report, ensure_ascii=False, indent=2),
                               file_name="report.json", mime="application/json", use_container_width=True)

        # Also save to file
        with open("best_prompt.txt", "w", encoding="utf-8") as f:
            f.write(fs['best_prompt'])

# ══════════════════════════════════════════════
# TAB 5: History
# ══════════════════════════════════════════════
with tab_history:
    st.subheader("📜 История запусков")

    if not st.session_state.run_history:
        st.info("История пуста. Запустите пайплайн для создания записей.")
    else:
        for i, run in enumerate(reversed(st.session_state.run_history)):
            fs_h = run["final_state"]
            cfg_h = run["config"]
            ts = run.get("timestamp", "—")
            with st.expander(f"🕐 {ts} — Accuracy: {fs_h['best_accuracy']:.1f}% | {cfg_h['evaluator']}/{cfg_h['judge']}", expanded=(i == 0)):
                hc1, hc2, hc3 = st.columns(3)
                with hc1:
                    st.metric("Best Accuracy", f"{fs_h['best_accuracy']:.1f}%")
                with hc2:
                    st.metric("Итераций", fs_h.get('iteration', 1) - 1)
                with hc3:
                    st.metric("Train/Val", f"{cfg_h['train']}/{cfg_h['val']}")

                hist_metrics = fs_h.get('metrics_history', [])
                if hist_metrics:
                    st.dataframe(pd.DataFrame(hist_metrics), use_container_width=True, hide_index=True)

                run_logs = run.get("logs", [])
                if run_logs:
                    with st.expander("📋 Логи"):
                        st.code("\n".join(run_logs), language="log")
