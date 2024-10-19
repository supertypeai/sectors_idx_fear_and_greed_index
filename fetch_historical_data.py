import datetime
import os
import requests
import pandas as pd

from dotenv import load_dotenv
from supabase import create_client
from bs4 import BeautifulSoup


load_dotenv()

_SUPABASE_URL = os.getenv('SUPABASE_URL')
_SUPABASE_KEY = os.getenv('SUPABASE_KEY')
supabase_client = create_client(_SUPABASE_URL, _SUPABASE_KEY)

_EXCHANGE_RATE_API_KEY = os.getenv('EXCHANGE_RATE_API_KEY')

_TIMEZONE = 'UTC'

_MONTH_ENUM = {
    'januari': 1,
    'februari': 2,
    'maret': 3,
    'april': 4,
    'mei': 5,
    'juni': 6,
    'juli': 7,
    'agustus': 8,
    'september': 9,
    'oktober': 10,
    'november': 11,
    'desember': 12
}


def fetch_daily_data():
    date_1_y_ago = pd.Timestamp.now() - pd.Timedelta(365, 'days')

    response = (supabase_client.table('idx_daily_data')
                .select('symbol, date, close, volume, market_cap')
                .gte('date', date_1_y_ago.date())
                .execute())

    return pd.DataFrame(response.data)


def fetch_idr_usd_rate():
    rate_df = pd.read_csv('exchange_rate.csv')
    # convert date and timestamp to datetime type
    rate_df['date'] = pd.to_datetime(rate_df['date']).dt.date
    rate_df['timestamp'] = pd.to_datetime(rate_df['timestamp'])

    latest_date = rate_df['date'].max()
    today = pd.Timestamp.now(tz=_TIMEZONE).date()

    new_historical_data: list[dict] = []
    one_day = pd.Timedelta(1, 'days')
    current_date = today

    while current_date > latest_date:
        url = (
            f'https://openexchangerates.org/api/historical/{current_date}.json'
            f'?app_id={_EXCHANGE_RATE_API_KEY}'
            '&base=USD'
            '&symbols=IDR'
            )
        print(url)
        response = requests.get(url)
        data = response.json()
        new_record = {
            'date': current_date,
            'rate': data['rates']['IDR'],
            'timestamp': pd.Timestamp.fromtimestamp(float(data['timestamp']), _TIMEZONE)
        }
        new_historical_data.append(new_record)
        current_date = current_date - one_day

    new_rate_df = pd.DataFrame.from_records(new_historical_data)

    max_retention = today - pd.Timedelta(365, 'days')
    rate_df = rate_df.loc[rate_df['date'] > max_retention]
    rate_df = pd.concat([new_rate_df, rate_df], sort=True)
    rate_df.to_csv('exchange_rate.csv', index=False)

    return rate_df


def _id_date_format_to_datetime(date: str):
    split_date = date.split()
    return pd.Timestamp(
        year=int(split_date[2]),
        month=_MONTH_ENUM[split_date[1].lower()],
        day=int(split_date[0])
    ).date()


def fetch_idr_interest_rate():
    interest_df = pd.read_csv('interest_rate.csv')
    # convert date and timestamp to datetime type
    interest_df['date'] = pd.to_datetime(interest_df['date']).dt.date

    latest_date: datetime.date = interest_df['date'].max()
    today = pd.Timestamp.now(tz=_TIMEZONE).date()

    new_historical_data: list[dict] = []
    current_date = today

    if current_date.month > latest_date.month:
        url = 'https://www.bi.go.id/id/statistik/indikator/BI-Rate.aspx'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'lxml')
        table = soup.find('table')
        entries = table.find_all('tr')
        new_entries: list[dict] = []
        for entry in entries:
            cols = entry.find_all_next('td')
            date = cols[0].getText()
            rate = cols[1].getText()
            new_entries.append({
                'date': _id_date_format_to_datetime(date),
                'rate': float(rate.split()[0])
            })

        new_interest_df = pd.DataFrame.from_records(new_entries)

        # max_retention = today - pd.Timedelta(2 * 365, 'days')
        # interest_df = interest_df.loc[interest_df['date'] > max_retention]
        interest_df = pd.concat([new_interest_df, interest_df], sort=True).drop_duplicates()
        interest_df.to_csv('interest_rate.csv', index=False)

    return interest_df


if __name__ == '__main__':
    rate_df = fetch_idr_usd_rate()
    interest_df = fetch_idr_interest_rate()
