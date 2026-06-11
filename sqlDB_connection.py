import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()


def get_engine():
    db_url = os.getenv("DB_URL")
    if not db_url:
        raise EnvironmentError("DB_URL is not set. Add it to your .env file.")
    return create_engine(db_url)


def upload_dataframe(df: pd.DataFrame, table_name: str = "uci_heart_disease") -> None:
    engine = get_engine()
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"Uploaded {len(df)} rows to '{table_name}'.")


def query(sql: str) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)
