import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib

# Загружаем scaler и порядок фичей
scaler = joblib.load("models/scaler.joblib")
feature_order = joblib.load("models/features.joblib")


# ======================================================
# 1. Определяем архитектуру (такая же, как при обучении)
# ======================================================
class TitanicNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ======================================================
# 2. Загрузка обученной модели
# ======================================================
# путь к модели (в корне проекта)
model_path = Path("models/titanic_model.pth")

# создаём модель и загружаем веса
input_dim = 14 - 1  # столько признаков у нас без колонки Survived
model = TitanicNN(input_dim)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# ======================================================
# 3. Настройка страницы Streamlit
# ======================================================
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

st.title("🚢 Проверь, выжил бы ты на Титанике")
st.markdown(
    "Введите данные о себе и проверьте, какова была бы вероятность вашего спасения "
    "на борту легендарного Титаника 🧊"
)
st.divider()

# ======================================================
# 4. Форма для ввода данных
# ======================================================

col1, col2 = st.columns(2)

with col1:
    sex = st.selectbox("Пол", options=["Мужчина", "Женщина"])
    age = st.slider("Возраст", 0, 80, 30)
    sibsp = st.number_input("Братья/сёстры на борту", 0, 8, 0)
    parch = st.number_input("Родители/дети на борту", 0, 6, 0)
    fare = st.number_input("Стоимость билета (£)", 0.0, 600.0, 30.0, step=1.0)

with col2:
    pclass = st.radio("Класс каюты", options=[1, 2, 3], horizontal=True)
    embarked = st.selectbox("Порт посадки", options=["C (Шербур)", "Q (Куинстаун)", "S (Саутгемптон)"])

st.divider()

# ======================================================
# 5. Подготовка признаков к подаче в модель
# ======================================================

# Пол
sex = 1 if sex == "Женщина" else 0

# Порты
embarked_C = 1 if embarked.startswith("C") else 0
embarked_Q = 1 if embarked.startswith("Q") else 0
embarked_S = 1 if embarked.startswith("S") else 0

# Классы
pclass_1 = 1 if pclass == 1 else 0
pclass_2 = 1 if pclass == 2 else 0
pclass_3 = 1 if pclass == 3 else 0

# Дополнительные фичи (как в ноутбуке)
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

# Формируем DataFrame с признаками
user_data = {
    "Sex": [sex],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "FamilySize": [family_size],
    "IsAlone": [is_alone],
    "Pclass_1": [pclass_1],
    "Pclass_2": [pclass_2],
    "Pclass_3": [pclass_3],
    "Embarked_C": [embarked_C],
    "Embarked_Q": [embarked_Q],
    "Embarked_S": [embarked_S],
}

user_df = pd.DataFrame(user_data)

# Масштабируем нужные признаки с помощью сохранённого scaler
cols_to_scale = ["Age", "Fare", "SibSp", "Parch"]
user_df[cols_to_scale] = scaler.transform(user_df[cols_to_scale])

# Упорядочиваем колонки в том же порядке, как при обучении
user_df = user_df[feature_order]

# Превращаем в тензор
X_tensor = torch.tensor(user_df.values, dtype=torch.float32)

# ======================================================
# 6. Кнопка предсказания
# ======================================================
if st.button("🧮 Предсказать вероятность выживания"):
    with torch.no_grad():
        prob = model(X_tensor).item()
        prob_percent = prob * 100

    # Формируем вывод
    if prob > 0.75:
        verdict = "🟢 Вы бы **скорее всего выжили** вместе с пассажирами первого класса 👑"
    elif prob > 0.5:
        verdict = "🟡 Ваши шансы 50/50 — всё зависело бы от удачи и места на корабле ⚖️"
    else:
        verdict = "🔴 Увы, вы **вряд ли бы выжили**... спасательные шлюпки закончились 😢"

    st.subheader(f"💡 Вероятность выживания: **{prob_percent:.1f}%**")
    st.markdown(verdict)
