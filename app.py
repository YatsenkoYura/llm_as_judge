import streamlit as st
import pandas as pd
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from dataset import process_dataset
from graph import build_workflow
from models import GraphState

# Load default environment variables
load_dotenv()

st.set_page_config(page_title="LLM Judge Optimizer", layout="wide")

st.title("LLM Judge Optimizer")
st.markdown("Настройте параметры, загрузите датасет, разметьте столбцы и запустите пайплайн поиска лучшего системного промпта для судьи.")

with st.sidebar:
    st.header("1. Конфигурация API")
    api_token = st.text_input("OpenAI API Token", value=os.getenv("API_TOKEN", ""), type="password")
    api_base_url = st.text_input("API Base URL", value=os.getenv("API_BASE_URL", "https://api.openai.com/v1"))
    evaluator_model = st.text_input("Evaluator Model", value=os.getenv("EVALUATOR_MODEL", "gemini-3-flash-preview"))
    judge_model = st.text_input("Judge Model", value=os.getenv("JUDGE_MODEL", "gemini-3-flash-preview"))
    
    st.header("2. Параметры пайплайна")
    default_task = "Оцени диалог. 1 - агент решил задачу, 0 - не решил."
    task_desc = st.text_area("Описание задачи", value=default_task)
    
    num_train = st.number_input("Размер обучающей выборки (Train)", min_value=1, value=10)
    num_val = st.number_input("Размер валидационной выборки (Val)", min_value=1, value=20)
    max_iter = st.number_input("Макс. количество итераций", min_value=1, value=3)
    target_acc = st.number_input("Целевая точность (%)", min_value=1.0, max_value=100.0, value=90.0)

st.header("3. Данные")
uploaded_file = st.file_uploader("Загрузите датасет", type=["csv", "json"])

if uploaded_file is not None:
    # Определение формата и загрузка данных
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_json(uploaded_file)
            
        st.write("Превью данных:")
        st.dataframe(df.head(3))
        
        columns = df.columns.tolist()
        
        st.subheader("Разметка столбцов (маппинг в UniversalExample)")
        col1, col2, col3 = st.columns(3)
        with col1:
            context_col = st.selectbox("Столбец контекста (Диалог/Логи)", options=columns)
            drop_duplicates = st.checkbox("Удалить дубликаты диалогов", value=True, help="Полезно для датасетов вроде DICES, где один диалог оценивается разными асессорами несколько раз.")
            
        with col2:
            st.markdown("**Настройка столбцов меток**")
            label_mode = st.radio("Режим объединения:", ["Один столбец", "Несколько столбцов (любое совпадение -> 1)"])
            if label_mode == "Один столбец":
                label_cols = [st.selectbox("Столбец ожидаемой оценки", options=columns)]
            else:
                label_cols = st.multiselect("Столбцы ожидаемой оценки", options=columns)
                
        with col3:
            explanation_col = st.selectbox("Столбец объяснения/логики разметки", options=columns)
            
        st.subheader("Настройки маппинга меток")
        map_col1, map_col2, map_col3 = st.columns(3)
        with map_col1:
            val_for_1_str = st.text_input("Значения для метки 1 (через запятую)", value="Yes, 1, True")
            val_for_1 = [v.strip() for v in val_for_1_str.split(',')]
        with map_col2:
            val_for_0_str = st.text_input("Значения для метки 0 (через запятую)", value="No, 0, False")
            val_for_0 = [v.strip() for v in val_for_0_str.split(',')]
        with map_col3:
            case_sensitive = st.checkbox("Чувствительность к регистру", value=False)
            
        if st.button("🚀 Запустить пайплайн", type="primary"):
            if not api_token:
                st.error("Пожалуйста, введите API Token в боковой панели.")
            else:
                with st.spinner("Запуск пайплайна... Смотрите логи в консоли сервера или дождитесь завершения."):
                    # Подготовка данных
                    if not label_cols:
                        st.error("Пожалуйста, выберите хотя бы один столбец ожидаемой оценки.")
                        st.stop()
                        
                    if drop_duplicates:
                        df = df.drop_duplicates(subset=[context_col])
                        
                    train_data, val_data = process_dataset(
                        df=df,
                        context_col=context_col,
                        label_cols=label_cols,
                        explanation_col=explanation_col,
                        num_train=num_train,
                        num_val=num_val,
                        val_for_1=val_for_1,
                        val_for_0=val_for_0,
                        case_sensitive=case_sensitive
                    )
                    
                    if len(train_data) == 0 or len(val_data) == 0:
                        st.error("Ошибка: Обучающая или валидационная выборка пуста. Проверьте размер датасета.")
                    else:
                        # Настройка клиента
                        client = OpenAI(
                            api_key=api_token,
                            base_url=api_base_url if api_base_url else None
                        )
                        
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
                        
                        graph = build_workflow(client, evaluator_model, judge_model)
                        
                        # Запуск графа
                        final_state = graph.invoke(initial_state)
                        
                        st.success("Пайплайн успешно завершен!")
                        
                        st.header("🏆 Результаты")
                        st.metric("Лучшая точность", f"{final_state['best_accuracy']:.1f}%")
                        
                        st.subheader("Лучший промпт")
                        st.code(final_state['best_prompt'], language="markdown")
                        
                        # Сохранение в файл
                        with open("best_prompt.txt", "w", encoding="utf-8") as f:
                            f.write(final_state['best_prompt'])
                        st.info("Лучший промпт сохранен в 'best_prompt.txt'")
                        
    except Exception as e:
        st.error(f"Ошибка при обработке файла: {str(e)}")
else:
    st.info("Пожалуйста, загрузите файл CSV или JSON для начала работы.")
