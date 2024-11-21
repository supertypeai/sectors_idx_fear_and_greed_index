import datetime
import locale
import os
import requests
import csv
import pandas as pd

from contextlib import contextmanager
from dotenv import load_dotenv
from supabase import create_client
from bs4 import BeautifulSoup

load_dotenv()

_SUPABASE_URL = os.getenv('SUPABASE_URL')
_SUPABASE_KEY = os.getenv('SUPABASE_KEY')
supabase_client = create_client(_SUPABASE_URL, _SUPABASE_KEY)

_SECTORS_API_KEY = os.getenv('SECTORS_API_KEY')
_EXCHANGE_RATE_API_KEY = os.getenv('EXCHANGE_RATE_API_KEY')

_TIMEZONE = 'UTC'
_LOCAL_TIMEZONE = 'Asia/Bangkok'

_ONE_WEEK = 11
_ONE_YEAR = 365
_TWO_YEARS = 2 * 365

_SCRAPING_HEADER = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br, zstd',
    'Connection': 'keep-alive'
}

_INDOBEX_NAME_ENUM = {
    'Indonesia Composite Bond Index (ICBI)': 'ICBI',
    'INDOBeX Composite Clean Price': 'INDOBeX-CP',
    'INDOBeX Composite Gross Price': 'INDOBeX-GP',
    'INDOBeX Composite Effective Yield': 'INDOBeX-EY',
    'INDOBeX Composite Gross Yield': 'INDOBeX-GY',
}

_DAILY_DATA_PAGE_SIZE = 90
_MCAP_DATA_PAGE_SIZE = 90

@contextmanager
def override_locale(category, locale_string):
    prev_locale_string = locale.getlocale(category)
    locale.setlocale(category, locale_string)
    yield
    locale.setlocale(category, prev_locale_string)


def get_previous_workday(date: datetime.date):
    date_weekday = date.weekday()
    if date_weekday == 0:  # monday
        day_delta = 3
    elif date_weekday == 6:  # sunday
        day_delta = 2
    else:
        day_delta = 1
    return date - datetime.timedelta(day_delta)


def fetch_daily_data(timeframe: int = _ONE_WEEK):
    idx30_csv_url = ('https://raw.githubusercontent.com/supertypeai/sectors_indices_company_list/main/company_list'
                     '/companies_list_idx30.csv')
    response = requests.get(idx30_csv_url)
    response.raise_for_status()

    csv_content = response.content.decode('utf-8')

    # Use csv.reader to parse the CSV content
    csv_reader = csv.reader(csv_content.splitlines(), delimiter=',')

    # Convert CSV to a list of tickers
    idx30_tickers = [row[0] for row in csv_reader]
    idx30_tickers = idx30_tickers[1:]

    records: list[dict] = []
    current_date = pd.Timestamp.now()
    remaining_timeframe = timeframe
    while remaining_timeframe > 0:
        timedelta = _DAILY_DATA_PAGE_SIZE if remaining_timeframe >= _DAILY_DATA_PAGE_SIZE else remaining_timeframe
        max_past_date = current_date - pd.Timedelta(timedelta, 'days')

        response = (supabase_client.table('idx_daily_data')
                    .select('symbol, date, close, volume, market_cap')
                    .gte('date', max_past_date.date())
                    .lte('date', current_date.date())
                    .in_('symbol', idx30_tickers)
                    .execute())
        records = response.data + records

        current_date = max_past_date - pd.Timedelta(1, 'days')
        remaining_timeframe -= timedelta

    daily_data = pd.DataFrame(records)
    daily_data['date'] = pd.to_datetime(daily_data['date']).dt.date
    # Assuming that data is sorted by date ascending from DB, no need to sort
    return daily_data


def fetch_mcap_data(timeframe: int = _ONE_WEEK):
    url = 'https://api.sectors.app/v1/idx-total/'
    headers = {
        'Authorization': _SECTORS_API_KEY,
    }

    records: list[dict] = []
    current_date = pd.Timestamp.now()
    remaining_timeframe = timeframe
    while remaining_timeframe > 0:
        timedelta = _MCAP_DATA_PAGE_SIZE if remaining_timeframe >= _MCAP_DATA_PAGE_SIZE else remaining_timeframe
        max_past_date = current_date - pd.Timedelta(timedelta, 'days')
        params = {
            'start': max_past_date.date().strftime('%Y-%m-%d'),
            'end': current_date.date().strftime('%Y-%m-%d')
        }

        response = requests.get(url, params=params, headers=headers)

        # Append the new data into the records
        data = response.json()
        records = data + records

        # Decrease the date and the remaining timeframe
        current_date = max_past_date - pd.Timedelta(1, 'days')
        remaining_timeframe -= timedelta

    # Combine the records into a dataframe
    mcap_data = pd.DataFrame.from_records(records)
    mcap_data['date'] = pd.to_datetime(mcap_data['date']).dt.date
    # Assuming data is sorted by date ascending, no need to sort
    return mcap_data


def fetch_idr_usd_rate(timeframe: int = _ONE_WEEK):
    rate_df = pd.read_csv('data/exchange_rate.csv')
    # convert date and timestamp to datetime type
    rate_df['date'] = pd.to_datetime(rate_df['date']).dt.date
    rate_df['timestamp'] = pd.to_datetime(rate_df['timestamp'])

    latest_date = rate_df['date'].max()
    today = pd.Timestamp.now(tz=_TIMEZONE).date()

    new_historical_data: list[dict] = []
    one_day = pd.Timedelta(1, 'days')
    current_date = today

    while current_date >= latest_date:
        url = f'https://openexchangerates.org/api/historical/{current_date}.json'
        params = {
            'app_id': _EXCHANGE_RATE_API_KEY,
            'base': 'USD',
            'symbols': 'IDR'
        }
        print(url)
        response = requests.get(url, params=params)
        data = response.json()
        new_record_timestamp = pd.Timestamp.fromtimestamp(float(data['timestamp']), _TIMEZONE)
        new_record = {
            'date': new_record_timestamp.tz_convert(_LOCAL_TIMEZONE).date(),
            'rate': data['rates']['IDR'],
            'timestamp': new_record_timestamp
        }
        new_historical_data.append(new_record)
        current_date = current_date - one_day

    new_rate_df = pd.DataFrame.from_records(new_historical_data)

    max_retention = today - pd.Timedelta(_TWO_YEARS, 'days')
    rate_df = rate_df.loc[rate_df['date'] > max_retention]
    rate_df = pd.concat([new_rate_df, rate_df], sort=True).drop_duplicates(subset=['date'], keep='first')
    rate_df.to_csv('data/exchange_rate.csv', index=False)

    max_past_date = today - pd.Timedelta(timeframe, 'days')
    return rate_df.loc[rate_df['date'] > max_past_date]


def _id_date_format_to_datetime(date: str):
    parsed_datetime = datetime.datetime.strptime(date, '%d %B %Y')
    return parsed_datetime.date()


def fetch_idr_interest_rate(timeframe: int = _ONE_YEAR):
    interest_df = pd.read_csv('data/interest_rate.csv')
    # convert date and timestamp to datetime type
    interest_df['date'] = pd.to_datetime(interest_df['date']).dt.date

    latest_date: datetime.date = interest_df['date'].max()
    today = pd.Timestamp.now(tz=_LOCAL_TIMEZONE).date()

    current_date = today

    if current_date.month > latest_date.month:
        url = 'https://www.bi.go.id/id/statistik/indikator/BI-Rate.aspx'
        print(url)
        response = requests.get(url, headers=_SCRAPING_HEADER)
        soup = BeautifulSoup(response.text, 'lxml')
        table = soup.find('table')
        tbody = table.find('tbody')
        entries = tbody.find_all('tr')
        new_entries: list[dict] = []
        with override_locale(locale.LC_TIME, 'id_ID.utf8'):
            for entry in entries:
                cols = entry.find_all('td')
                date = cols[0].getText()
                rate = cols[1].getText()
                new_entries.append({
                    'date': _id_date_format_to_datetime(date),
                    'rate': float(rate.split()[0])
                })

        new_interest_df = pd.DataFrame.from_records(new_entries)

        max_retention = today - pd.Timedelta(_TWO_YEARS, 'days')
        interest_df = interest_df.loc[interest_df['date'] > max_retention]
        interest_df = (pd.concat([new_interest_df, interest_df], sort=True)
                       .drop_duplicates(subset=['date'], keep='first'))
        interest_df.to_csv('data/interest_rate.csv', index=False)

    # Add additional +45 days of padding to include previous month in case of long interval
    max_past_date = today - pd.Timedelta(timeframe + 45, 'days')
    return interest_df.loc[interest_df['date'] > max_past_date]


def fetch_bonds_rate(timeframe: int = _ONE_WEEK, force_refresh=False):
    # Fetch Indonesian Bond Index
    bonds_df = pd.read_csv('data/bonds_rate.csv')
    # convert date and timestamp to datetime type
    bonds_df['date'] = pd.to_datetime(bonds_df['date']).dt.date

    latest_date: datetime.date = bonds_df['date'].max()
    today = pd.Timestamp.now(tz=_LOCAL_TIMEZONE).date()

    if today > latest_date or force_refresh:
        url = 'https://www.phei.co.id/Data/Indeks'
        print(url)

        response = requests.get(url, headers=_SCRAPING_HEADER)
        soup = BeautifulSoup(response.text, 'lxml')
        table = soup.find('table', attrs={'id': 'dnn_ctr1478_BondIndexes_indobexdata_gvDailyDate'})
        entries = table.find_all('tr')

        phei_date_string = soup.find('span', {'id': 'dnn_ctr1478_BondIndexes_indobexdata_lblDate'}).getText()
        with override_locale(locale.LC_TIME, 'id_ID.utf8'):
            phei_date = _id_date_format_to_datetime(phei_date_string)
        phei_date_previous = get_previous_workday(phei_date)

        new_entries: list[dict] = [
            {
                'date': phei_date
            },
            {
                'date': phei_date_previous
            }
        ]
        for entry in entries:
            cols = entry.find_all_next('td')
            if len(cols) == 0:
                continue
            name = cols[2].getText()
            yesterday_rate = (cols[3].getText()).replace(',', '.')
            latest_rate = (cols[4].getText()).replace(',', '.')
            try:
                col = _INDOBEX_NAME_ENUM[name]
                new_entries[0][col] = float(latest_rate)
                new_entries[1][col] = float(yesterday_rate)
            except KeyError:
                continue

        new_entry = pd.DataFrame.from_records(new_entries)

        max_retention = today - pd.Timedelta(_TWO_YEARS, 'days')
        bonds_df = bonds_df.loc[bonds_df['date'] > max_retention]
        interest_df = pd.concat([new_entry, bonds_df], sort=True).drop_duplicates(subset=['date'], keep='first')
        interest_df.to_csv('data/bonds_rate.csv', index=False)

    max_past_date = today - pd.Timedelta(timeframe, 'days')
    return bonds_df.loc[bonds_df['date'] > max_past_date]


def fetch_temp_bonds_rate(timeframe: int = _ONE_WEEK):
    # Fetch Indonesian Bond yield rate
    bonds_df = pd.read_csv('data/temp_bonds_rate.csv')
    # convert date and timestamp to datetime type
    bonds_df['date'] = pd.to_datetime(bonds_df['date']).dt.date

    latest_date: datetime.date = bonds_df['date'].max()
    today = pd.Timestamp.now(tz=_LOCAL_TIMEZONE).date()

    if today > latest_date:
        url = 'https://investing.com/rates-bonds/indonesia-10-year-bond-yield-historical-data'
        print(url)
        investingcom_header = _SCRAPING_HEADER.copy()
        investingcom_header['Host'] = 'www.investing.com'
        response = requests.get(url, headers=investingcom_header)
        soup = BeautifulSoup(response.text, 'lxml')
        table = soup.find('table')
        table_body = table.find('tbody')
        entries = table_body.find_all('tr')

        new_entries: list[dict] = []
        for entry in entries:
            cols = entry.find_all('td')
            date = datetime.datetime.strptime(cols[0].getText(), '%b %d, %Y').date()
            rate = cols[1].getText()
            new_entries.append({
                'date': date,
                'rate': float(rate)
            })

        new_bonds_df = pd.DataFrame.from_records(new_entries)

        max_retention = today - pd.Timedelta(_TWO_YEARS, 'days')
        bonds_df = bonds_df.loc[bonds_df['date'] > max_retention]
        bonds_df = pd.concat([new_bonds_df, bonds_df], sort=True).drop_duplicates(subset=['date'], keep='first')
        bonds_df.to_csv('data/temp_bonds_rate.csv', index=False)

    max_past_date = today - pd.Timedelta(timeframe, 'days')
    return bonds_df.loc[bonds_df['date'] > max_past_date]


def fetch_ihsg_data(timeframe: int = _ONE_WEEK):
    url = 'https://api.sectors.app/v1/index-daily/ihsg/'
    headers = {
        'Authorization': _SECTORS_API_KEY,
    }

    records: list[dict] = []
    current_date = pd.Timestamp.now()
    remaining_timeframe = timeframe
    while remaining_timeframe > 0:
        timedelta = _MCAP_DATA_PAGE_SIZE if remaining_timeframe >= _MCAP_DATA_PAGE_SIZE else remaining_timeframe
        max_past_date = current_date - pd.Timedelta(timedelta, 'days')
        params = {
            'start': max_past_date.date().strftime('%Y-%m-%d'),
            'end': current_date.date().strftime('%Y-%m-%d')
        }

        response = requests.get(url, params=params, headers=headers)

        # Append the new data into the records
        data = response.json()
        records = data + records

        # Decrease the date and the remaining timeframe
        current_date = max_past_date - pd.Timedelta(1, 'days')
        remaining_timeframe -= timedelta

    # Combine the records into a dataframe
    ihsg_data = pd.DataFrame.from_records(records)
    ihsg_data['date'] = pd.to_datetime(ihsg_data['date']).dt.date
    # Assuming data is sorted by date ascending, no need to sort
    return ihsg_data


def push_to_db(fear_and_greed_index: pd.DataFrame, n_latest=1):
    # Get n_latest last data (since the data is sorted oldest-latest)
    latest_data: pd.DataFrame = fear_and_greed_index.iloc[-n_latest:]
    # Create the date column from the index into ISO format for db insert
    latest_data = latest_data.copy()
    latest_data['date'] = latest_data.index
    latest_data['date'] = latest_data['date'].apply(lambda x: x.isoformat())
    # Convert to records
    latest_data_dict = latest_data.to_dict(orient='records')
    # Insert into db
    supabase_client.table('idx_fear_and_greed').insert(latest_data_dict).execute()


if __name__ == '__main__':
    daily_df = fetch_daily_data()
    print(daily_df)
    mcap_df = fetch_mcap_data()
    print(mcap_df)
    rate_df = fetch_idr_usd_rate()
    print(rate_df)
    interest_df = fetch_idr_interest_rate()
    print(interest_df)
    bonds_df = fetch_bonds_rate()
    print(bonds_df)
    temp_bonds_df = fetch_temp_bonds_rate()
    print(temp_bonds_df)
    ihsg_data = fetch_ihsg_data()
    print(ihsg_data)
