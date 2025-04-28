import os

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
    


    DB_USER = os.getenv("DB_USER", "myuser")
    DB_PASS = os.getenv("DB_PASS", "mypassword")
    DB_HOST = os.getenv("DB_HOST", "localhost")        # ← service name
    DB_PORT = os.getenv("DB_PORT", "5433")      # ← container-internal port
    DB_NAME = os.getenv("DB_NAME", "mydatabase")

    SQLALCHEMY_DATABASE_URI = (
        f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False