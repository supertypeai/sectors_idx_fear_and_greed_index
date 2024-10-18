import os
import requests
import datetime
import pandas as pd

from dotenv import load_dotenv

load_dotenv()
_EXCHANGE_RATE_API_KEY = os.getenv('EXCHANGE_RATE_API_KEY')

_TIMEZONE = 'UTC'


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


def fetch_idr_interest_rate():
    pass


if __name__ == "__main__":
    rate_df = fetch_idr_usd_rate()
    interest_df = fetch_idr_interest_rate()
