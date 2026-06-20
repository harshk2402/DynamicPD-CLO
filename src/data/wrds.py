import pandas as pd
from dotenv import load_dotenv

from src.utils.connections import get_wrds_connection

load_dotenv()


class WRDSClient:
    def __init__(self):
        self._db = get_wrds_connection()

    def close(self):
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def query(self, sql: str) -> pd.DataFrame:
        return self._db.raw_sql(sql)

    def list_libraries(self) -> list[str]:
        return self._db.list_libraries() or []

    def list_tables(self, library: str) -> list[str]:
        return self._db.list_tables(library=library) or []

    def describe_table(self, library: str, table: str) -> pd.DataFrame:
        return self._db.describe_table(library=library, table=table)
