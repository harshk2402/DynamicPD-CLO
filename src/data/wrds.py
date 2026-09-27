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

    def query_chunks(self, sql: str, chunksize: int = 200_000):
        """Stream a query in chunks (server-side cursor) to avoid loading huge tables into memory at once."""
        return self._db.raw_sql(sql, chunksize=chunksize)

    def stream_query(self, sql: str, batch_size: int = 50_000):
        """True server-side streaming via a named psycopg2 cursor. Opens its own short-lived raw
        connection (separate from the interactive wrds.Connection) so the Postgres backend keeps
        the result set server-side and sends it in batches, instead of buffering it all client-side.
        Yields pandas DataFrame chunks of up to batch_size rows."""
        from src.utils.connections import get_wrds_raw_connection

        conn = get_wrds_raw_connection()
        try:
            cur = conn.cursor(name="dynamicpd_clo_stream")
            cur.itersize = batch_size
            cur.execute(sql)
            colnames = None
            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    break
                if colnames is None:
                    colnames = [d[0] for d in cur.description]
                yield pd.DataFrame(rows, columns=colnames)
            cur.close()
        finally:
            conn.close()

    def count_rows(self, schema: str, table: str) -> int:
        result = self.query(f"SELECT COUNT(*) AS n FROM {schema}.{table}")
        return int(result["n"].iloc[0])

    def list_libraries(self) -> list[str]:
        return self._db.list_libraries() or []

    def list_tables(self, library: str) -> list[str]:
        return self._db.list_tables(library=library) or []

    def describe_table(self, library: str, table: str) -> pd.DataFrame:
        return self._db.describe_table(library=library, table=table)
