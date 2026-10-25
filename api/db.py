import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker

load_dotenv()


def build_url() -> URL:
    return URL.create(
        "postgresql+psycopg2",
        username=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.environ["POSTGRES_HOST"],
        port=int(os.environ["POSTGRES_PORT"]),
        database=os.environ["POSTGRES_DB"],
    )


engine = create_engine(
    build_url(),
    pool_pre_ping=True,                 # test connections before using them
    connect_args={"connect_timeout": 3},  # fail fast if the database is down
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)