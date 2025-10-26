# 🚢 Titanic Survival Predictor

Интерактивный ML-проект на Python + PyTorch + Streamlit.  
Модель предсказывает вероятность того, **выжил бы человек на Титанике**, исходя из введённых характеристик (пол, возраст, класс каюты, стоимость билета и т.д.).

---

## 📦 Стек технологий
- Python 3.10+
- PostgreSQL + SQLAlchemy
- Pandas / NumPy / Scikit-learn
- PyTorch
- Streamlit
- dotenv / joblib

---

## 🚀 Запуск проекта с нуля

### 1️⃣ Склонируй репозиторий
```bash
git clone https://github.com/oNovalSliveDNo/titanic.git
cd titanic
````

### 2️⃣ Установи зависимости и активируй виртуальное окружение

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

---

### 3️⃣ Скачай данные

Зайди на [Kaggle Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
и скачай файл **`Titanic-Dataset.csv`** → помести его в папку:

```
/data/Titanic-Dataset.csv
```

---

### 4️⃣ Подготовь PostgreSQL

Убедись, что установлен и запущен PostgreSQL (по умолчанию):

```
user: postgres
password: postgres
host: localhost
port: 5432
```

Создай файл **`.env`** в корне проекта:

```env
PG_HOST=localhost
PG_PORT=5432
PG_USER=postgres
PG_PASSWORD=postgres
PG_DB=titanic_demo
```

---

### 5️⃣ Инициализация базы данных

```bash
python db/init_db.py
python db/insert_data.py
```

После этого:

* создастся база данных `titanic_demo`
* появятся схемы `raw` и `processed`
* в `raw` загрузится исходный CSV-датасет

---

### 6️⃣ Обработка и обучение модели

Открой Jupyter Notebook:

```bash
jupyter notebook
```

и последовательно выполни:

```
notebooks/preprocessing.ipynb
```

После выполнения:

* обработанные данные сохранятся в `processed.titanic_train` и `processed.titanic_test`
* обученная модель и трансформеры сохранятся в:

  ```
  /models/titanic_model.pth
  /models/scaler.joblib
  /models/features.joblib
  ```

---

### 7️⃣ Запуск Streamlit-приложения

```bash
streamlit run app.py
```

Откроется страница по адресу [http://localhost:8501](http://localhost:8501)

---

## 🧠 Интерфейс

Пользователь вводит параметры:

* Пол
* Возраст
* Кол-во родственников
* Класс каюты
* Порт посадки
* Стоимость билета

→ модель возвращает **вероятность выживания** и дружелюбный комментарий 💬

---

## 📂 Структура проекта

```
titanic/
├── app.py
├── .env
├── requirements.txt
├── data/
│   └── Titanic-Dataset.csv
├── db/
│   ├── db.py
│   ├── init_db.py
│   └── insert_data.py
├── models/
│   ├── titanic_model.pth
│   ├── scaler.joblib
│   └── features.joblib
├── notebooks/
│   ├── preprocessing.ipynb
└── README.md
```

---

## 💡 Пример вывода

```
💡 Вероятность выживания: 98.2 %
🟢 Вы бы скорее всего выжили вместе с пассажирами первого класса 👑
```
