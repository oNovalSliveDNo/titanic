import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv
import os

# Загружаем .env файл
load_dotenv()

# Получаем параметры подключения
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_DB = os.getenv("PG_DB")

# Создаём пул подключений (чтобы можно было безопасно делить соединения)
connection_pool = None


def init_pool():
    global connection_pool
    if connection_pool is None:
        connection_pool = psycopg2.pool.SimpleConnectionPool(
            1, 10,
            user=PG_USER,
            password=PG_PASSWORD,
            host=PG_HOST,
            port=PG_PORT,
            database=PG_DB
        )


def get_conn():
    """Получить соединение из пула"""
    if connection_pool is None:
        init_pool()
    return connection_pool.getconn()


def put_conn(conn):
    """Вернуть соединение в пул"""
    if connection_pool:
        connection_pool.putconn(conn)


def close_pool():
    """Закрыть пул"""
    if connection_pool:
        connection_pool.closeall()
