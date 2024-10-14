import os
import pandas as pd

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

_SUPABASE_URL = os.getenv("SUPABASE_URL")
_SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase_client = create_client(_SUPABASE_URL, _SUPABASE_KEY)


def fetch_daily_data():
    date_1_w_ago = pd.Timestamp.now() - pd.Timedelta(7, 'days')

    response = (supabase_client.table("idx_daily_data")
                .select("symbol, date, close, volume")
                .gte('date', date_1_w_ago.date())
                .execute())

    return pd.DataFrame(response.data)


def calculate_sma(data: pd.DataFrame):
    pass


def calculate_fear_and_greed_index(idx_daily_data: pd.DataFrame):
    pass
