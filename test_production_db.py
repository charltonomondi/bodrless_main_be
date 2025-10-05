import psycopg2
import os

# Use your Render database credentials
DB_NAME = os.getenv("DB_NAME", "bodrless_db")
DB_USER = os.getenv("DB_USER", "bodrless_db_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "2iz4viQHDOZNSwKRspgbfBqayATqcCwI")
DB_HOST = os.getenv("DB_HOST", "dpg-d3gh5uggjchc739p2ao0-a.oregon-postgres.render.com")
DB_PORT = os.getenv("DB_PORT", "5432")

try:
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    print("✅ Database connection successful!")
    cur = conn.cursor()
    cur.execute("SELECT version();")
    db_version = cur.fetchone()
    print("PostgreSQL version:", db_version)
    cur.close()
    conn.close()
except Exception as e:
    print("❌ Database connection failed!")
    print("Error:", e)
