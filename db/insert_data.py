import pandas as pd
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
import os

load_dotenv()

PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_DB = os.getenv("PG_DB")

DATA_PATH = os.path.join("data", "Titanic-Dataset.csv")


def infer_sql_type(dtype: str) -> str:
    """Преобразует dtype pandas в SQL тип"""
    if "int" in dtype:
        return "INTEGER"
    elif "float" in dtype:
        return "DOUBLE PRECISION"
    elif "bool" in dtype:
        return "BOOLEAN"
    else:
        return "TEXT"


def insert_data():
    """Создаёт таблицу raw.titanic и вставляет данные"""
    df = pd.read_csv(DATA_PATH)
    print(f"📊 Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")

    conn = psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        dbname=PG_DB
    )
    cur = conn.cursor()

    # Автоматически создаём SQL-таблицу под колонки CSV
    columns = []
    for col, dtype in df.dtypes.items():
        sql_type = infer_sql_type(str(dtype))
        columns.append(f'"{col}" {sql_type}')
    columns_sql = ", ".join(columns)

    create_query = f"""
    DROP TABLE IF EXISTS raw.titanic;
    CREATE TABLE raw.titanic (
        id SERIAL PRIMARY KEY,
        {columns_sql}
    );
    """
    cur.execute(create_query)
    conn.commit()
    print("✅ Table raw.titanic created.")

    # Вставляем данные
    for _, row in df.iterrows():
        cur.execute(
            sql.SQL("INSERT INTO raw.titanic ({}) VALUES ({})").format(
                sql.SQL(", ").join(map(sql.Identifier, df.columns)),
                sql.SQL(", ").join(sql.Placeholder() * len(df.columns))
            ),
            tuple(None if pd.isna(x) else x for x in row)
        )
    conn.commit()

    cur.close()
    conn.close()
    print("🚀 Data inserted successfully into raw.titanic.")


if __name__ == "__main__":
    insert_data()
