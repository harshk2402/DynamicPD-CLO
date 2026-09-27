from pathlib import Path
from typing import Callable

import pandas as pd
from dotenv import load_dotenv

from src.utils.connections import get_wrds_connection

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data/raw/merton"
CRSP_CHUNK_SIZE = 500


class DDDataLoader:
    def __init__(self, raw_dir: str | Path = RAW_DIR):
        self.raw_dir = Path(raw_dir).resolve()
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._db = get_wrds_connection()

    def close(self) -> None:
        self._db.close()

    def _fetch_or_load(
        self,
        filename: str,
        fetcher: Callable[[], pd.DataFrame],
        date_columns: tuple[str, ...] = (),
    ) -> pd.DataFrame:
        path = self.raw_dir / filename
        if path.exists():
            print(f"Cache hit: {path}")
            df = pd.read_parquet(path)
        else:
            print(f"Downloading: {filename}")
            df = fetcher()
            for column in date_columns:
                if column in df.columns:
                    df[column] = pd.to_datetime(df[column])
            df.to_parquet(path, index=False)
            print(f"Cached: {path} ({len(df):,} rows)")

        for column in date_columns:
            if column in df.columns:
                df[column] = pd.to_datetime(df[column])
        return df

    def load_ccm_links(self) -> pd.DataFrame:
        def fetch() -> pd.DataFrame:
            return self._db.raw_sql("""
                SELECT
                    gvkey,
                    lpermno,
                    lpermco,
                    linkdt,
                    linkenddt,
                    linktype,
                    linkprim
                FROM crsp.ccmxpf_lnkhist
                WHERE lpermno IS NOT NULL
                  AND linktype IN ('LC', 'LU')
                  AND linkprim IN ('P', 'C')
                """)

        df = self._fetch_or_load(
            "ccm_links.parquet",
            fetch,
            date_columns=("linkdt", "linkenddt"),
        )
        return df.loc[df["linktype"].isin(["LC", "LU"])].reset_index(drop=True)

    def load_compustat(self) -> pd.DataFrame:
        def fetch() -> pd.DataFrame:
            return self._db.raw_sql("""
                SELECT
                    gvkey,
                    datadate,
                    fyearq,
                    datafqtr,
                    datacqtr,
                    dlcq,
                    dlttq,
                    atq,
                    ltq,
                    tic,
                    conm
                FROM comp.fundq
                WHERE datadate BETWEEN '1999-01-01' AND '2024-12-31'
                  AND indfmt = 'INDL'
                  AND datafmt = 'STD'
                  AND popsrc = 'D'
                  AND consol = 'C'
                """)

        return self._fetch_or_load(
            "compustat_debt.parquet",
            fetch,
            date_columns=("datadate",),
        )

    def load_risk_free(self) -> pd.DataFrame:
        def fetch() -> pd.DataFrame:
            return self._db.raw_sql("""
                SELECT
                    date,
                    tb3ms
                FROM frb.rates_monthly
                WHERE date BETWEEN '1999-01-01' AND '2024-12-31'
                ORDER BY date
                """)

        return self._fetch_or_load(
            "tb3ms_monthly.parquet",
            fetch,
            date_columns=("date",),
        )

    def load_matched_links(
        self,
        ccm: pd.DataFrame | None = None,
        compustat: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if ccm is None:
            ccm = self.load_ccm_links()
        if compustat is None:
            compustat = self.load_compustat()

        def fetch() -> pd.DataFrame:
            links = ccm.copy()
            comp = compustat.copy()

            links["gvkey"] = links["gvkey"].astype(str)
            comp["gvkey"] = comp["gvkey"].astype(str)
            links["linkdt"] = pd.to_datetime(links["linkdt"]).fillna(
                pd.Timestamp("1900-01-01")
            )
            links["linkenddt"] = pd.to_datetime(links["linkenddt"]).fillna(
                pd.Timestamp("2099-12-31")
            )
            comp["datadate"] = pd.to_datetime(comp["datadate"])

            matched = comp.merge(links, on="gvkey", how="inner")
            matched = matched[
                (matched["datadate"] >= matched["linkdt"])
                & (matched["datadate"] <= matched["linkenddt"])
            ]
            return self._select_single_permno(matched)

        matched = self._fetch_or_load(
            "ccm_compustat_matched.parquet",
            fetch,
            date_columns=("datadate", "linkdt", "linkenddt"),
        )
        matched["linkdt"] = pd.to_datetime(matched["linkdt"]).fillna(
            pd.Timestamp("1900-01-01")
        )
        matched["linkenddt"] = pd.to_datetime(matched["linkenddt"]).fillna(
            pd.Timestamp("2099-12-31")
        )
        matched = matched.loc[matched["linktype"].isin(["LC", "LU"])]
        return self._select_single_permno(matched)

    def _select_single_permno(self, matched: pd.DataFrame) -> pd.DataFrame:
        matched = matched.copy()
        matched["linkprim_rank"] = matched["linkprim"].map({"P": 0, "C": 1}).fillna(9)
        matched["linktype_rank"] = matched["linktype"].map({"LC": 0, "LU": 1}).fillna(9)
        matched["lpermno"] = matched["lpermno"].astype(int)
        matched = matched.sort_values(
            [
                "gvkey",
                "datadate",
                "linkprim_rank",
                "linktype_rank",
                "linkdt",
                "lpermno",
            ],
            ascending=[True, True, True, True, False, True],
        )
        matched = matched.drop_duplicates(["gvkey", "datadate"], keep="first")
        return matched.drop(columns=["linkprim_rank", "linktype_rank"]).reset_index(
            drop=True
        )

    def valid_permnos(self, matched_links: pd.DataFrame) -> list[int]:
        permnos = sorted(
            matched_links["lpermno"].dropna().astype(int).unique().tolist()
        )

        if not permnos:
            raise ValueError("No valid PERMNOs produced from Compustat/CCM match.")

        print(f"Valid linked PERMNOs: {len(permnos):,}")
        return permnos

    def load_crsp_daily(
        self,
        matched_links: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        path = self.raw_dir / "crsp_daily.parquet"
        if path.exists():
            print(f"Cache hit: {path}")
            df = pd.read_parquet(path)
            df["date"] = pd.to_datetime(df["date"])
            return df

        if matched_links is None:
            matched_links = self.load_matched_links()

        permnos = self.valid_permnos(matched_links)
        frames: list[pd.DataFrame] = []
        total_chunks = (len(permnos) + CRSP_CHUNK_SIZE - 1) // CRSP_CHUNK_SIZE

        for chunk_number, start in enumerate(
            range(0, len(permnos), CRSP_CHUNK_SIZE), 1
        ):
            chunk = permnos[start : start + CRSP_CHUNK_SIZE]
            permno_sql = ", ".join(str(permno) for permno in chunk)
            print(
                f"Downloading CRSP daily chunk {chunk_number}/{total_chunks} "
                f"({len(chunk):,} PERMNOs)"
            )
            frames.append(self._db.raw_sql(f"""
                    SELECT
                        permno,
                        date,
                        prc,
                        shrout,
                        ret,
                        retx
                    FROM crsp.dsf
                    WHERE date BETWEEN '1998-12-01' AND '2024-12-31'
                      AND permno IN ({permno_sql})
                    """))

        df = pd.concat(frames, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["permno", "date"]).reset_index(drop=True)
        df.to_parquet(path, index=False)
        print(f"Cached: {path} ({len(df):,} rows)")
        return df

    def load_all(self) -> dict[str, pd.DataFrame]:
        ccm = self.load_ccm_links()
        compustat = self.load_compustat()
        matched_links = self.load_matched_links(ccm=ccm, compustat=compustat)
        risk_free = self.load_risk_free()
        crsp_daily = self.load_crsp_daily(matched_links=matched_links)
        datasets = {
            "ccm_links": ccm,
            "compustat_debt": compustat,
            "ccm_compustat_matched": matched_links,
            "tb3ms_monthly": risk_free,
            "crsp_daily": crsp_daily,
        }

        print("Final dataset sizes:")
        for name, df in datasets.items():
            print(f"  {name}: {len(df):,} rows")

        return datasets


def main() -> None:
    loader = DDDataLoader()
    try:
        loader.load_all()
    finally:
        loader.close()


if __name__ == "__main__":
    main()
