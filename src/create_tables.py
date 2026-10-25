import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # so "api" can be imported

from sqlalchemy import inspect

from api.db import engine
from api.db_models import Base

Base.metadata.create_all(engine)  # creates only tables that don't exist yet
print("Tables in database:", inspect(engine).get_table_names())