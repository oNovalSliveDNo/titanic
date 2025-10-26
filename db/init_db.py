import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv
import os

load_dotenv()

PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_DB = os.getenv("PG_DB")


def create_database():
    """Создаёт БД, если не существует"""
    conn = psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{PG_DB}';")
    exists = cur.fetchone()

    if not exists:
        cur.execute(f'CREATE DATABASE {PG_DB};')
        print(f"✅ Database '{PG_DB}' created successfully.")
    else:
        print(f"ℹ️ Database '{PG_DB}' already exists.")

    cur.close()
    conn.close()


def create_schemas():
    """Создаёт схемы raw и processed"""
    conn = psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        dbname=PG_DB
    )
    cur = conn.cursor()

    for schema in ["raw", "processed"]:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")
        print(f"✅ Schema '{schema}' ready.")

    conn.commit()
    cur.close()
    conn.close()


if __name__ == "__main__":
    create_database()
    create_schemas()
    print("🎯 Database initialization completed.")
