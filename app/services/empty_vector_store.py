# psycopg2 version
import os
import psycopg2
from psycopg2 import sql

def clear_pgvector_table(table_name: str):
    """
    Connects to Postgres and truncates the specified pgvector table.
    """
    conn = psycopg2.connect(
        host     = os.getenv("PGHOST", "localhost"),
        port     = os.getenv("PGPORT", "5433"),
        dbname   = os.getenv("PGDATABASE", "mydatabase"),
        user     = os.getenv("PGUSER", "myuser"),
        password = os.getenv("PGPASSWORD", "mypassword"),
    )
    try:
        with conn.cursor() as cur:
            # Remove all rows quickly; CASCADE if you have FK deps
            cur.execute(sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY CASCADE;")
                        .format(sql.Identifier(table_name)))
        conn.commit()
        print(f"🗑️ Successfully cleared table '{table_name}'.")
    finally:
        conn.close()

if __name__ == "__main__":
    clear_pgvector_table("table_chunks\d")
