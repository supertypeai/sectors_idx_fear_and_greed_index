import datetime
import os
import requests
import csv
# import cfscrape
import pandas as pd

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
    'desember': 12,

    'january': 1,
    'february': 2,
    'march': 3,
    # 'april': 4,
    'may': 5,
    'june': 6,
    'july': 7,
    'august': 8,
    # 'september': 9,
    'october': 10,
    # 'november': 11,
    'december': 12
}

_INDOBEX_NAME_ENUM = {
    'Indonesia Composite Bond Index (ICBI)': 'ICBI',
    'INDOBeX Composite Clean Price': 'INDOBeX-CP',
    'INDOBeX Composite Gross Price': 'INDOBeX-GP',
    'INDOBeX Composite Effective Yield': 'INDOBeX-EY',
    'INDOBeX Composite Gross Yield': 'INDOBeX-GY',
}


def fetch_daily_data():
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

    date_90_d_ago = pd.Timestamp.now() - pd.Timedelta(90, 'days')

    response = (supabase_client.table('idx_daily_data')
                .select('symbol, date, close, volume, market_cap')
                .gte('date', date_90_d_ago.date())
                .in_('symbol', idx30_tickers)
                .execute())

    return pd.DataFrame(response.data)


def fetch_mcap_data():
    url = 'https://api.sectors.app/v1/idx-total/'

    today = pd.Timestamp.now()
    date_90_d_ago = today - pd.Timedelta(90, 'days')

    headers = {
        'Authorization': _SECTORS_API_KEY,
    }
    params = {
        'start': date_90_d_ago.date().strftime('%Y-%m-%d'),
        'end': today.date().strftime('%Y-%m-%d')
    }

    response = requests.get(url, params=params, headers=headers)

    data = response.json()
    return pd.DataFrame.from_records(data)


def fetch_idr_usd_rate():
    rate_df = pd.read_csv('data/exchange_rate.csv')
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
    rate_df.to_csv('data/exchange_rate.csv', index=False)

    return rate_df


def _id_date_format_to_datetime(date: str):
    split_date = date.split()
    return pd.Timestamp(
        year=int(split_date[2]),
        month=_MONTH_ENUM[split_date[1].lower()],
        day=int(split_date[0])
    ).date()


def fetch_idr_interest_rate():
    interest_df = pd.read_csv('data/interest_rate.csv')
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
        interest_df.to_csv('data/interest_rate.csv', index=False)

    return interest_df


def fetch_bonds_rate(force_refresh=False):
    # Fetch Indonesian Bond Index
    bonds_df = pd.read_csv('data/bonds_rate.csv')
    # convert date and timestamp to datetime type
    bonds_df['date'] = pd.to_datetime(bonds_df['date']).dt.date

    latest_date: datetime.date = bonds_df['date'].max()
    today = pd.Timestamp.now(tz=_LOCAL_TIMEZONE).date()

    if today > latest_date or force_refresh:
        url = 'https://www.phei.co.id/Data/Indeks'
        print(url)

        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'lxml')
        table = soup.find('table', attrs={'id': 'dnn_ctr1478_BondIndexes_indobexdata_gvDailyDate'})
        entries = table.find_all('tr')

        phei_date_string = soup.find('span', {'id': 'dnn_ctr1478_BondIndexes_indobexdata_lblDate'}).getText()
        phei_date = _id_date_format_to_datetime(phei_date_string)
        phei_date_previous = phei_date - pd.Timedelta(1, 'days')

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

        # max_retention = today - pd.Timedelta(2 * 30, 'days')
        # bonds_df = bonds_df.loc[bonds_df['date'] > max_retention]
        interest_df = pd.concat([new_entry, bonds_df], sort=True).drop_duplicates()
        interest_df.to_csv('data/bonds_rate.csv', index=False)

    return bonds_df


def fetch_temp_bonds_rate():
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

        # max_retention = today - pd.Timedelta(2 * 30, 'days')
        # bonds_df = bonds_df.loc[bonds_df['date'] > max_retention]
        interest_df = pd.concat([new_bonds_df, bonds_df], sort=True).drop_duplicates()
        interest_df.to_csv('data/temp_bonds_rate.csv', index=False)

    return bonds_df


if __name__ == '__main__':
    # rate_df = fetch_idr_usd_rate()
    # interest_df = fetch_idr_interest_rate()
    bonds_rate = fetch_bonds_rate()
    temp_bonds_rate = fetch_temp_bonds_rate()
