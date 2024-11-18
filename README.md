# IDX Fear and Greed Index
Calculates the Fear and Greed Index for Indonesian stock market.

## Data Source
* Daily data: Primary Database
* Market cap: Sectors API (https://api.sectors.app/v1/idx-total/)
* IDR-USD rate: openexchangerates.org (https://openexchangerates.org/api/historical/{current_date}.jsonhttps://openexchangerates.org/api/historical/{date}.json)
* IDR interest rate: Bank Indonesia (https://www.bi.go.id/id/statistik/indikator/BI-Rate.aspx)
* Indonesian bonds(not yet used): PHEI (https://www.phei.co.id/Data/Indeks)
* Indonesian bonds yield (temporary): investing.com (https://investing.com/rates-bonds/indonesia-10-year-bond-yield-historical-data)

## Data Processing
Data are stored locally in CSV format at [data directory](./data), and are updated periodically along with the daily calculation.
Data fetching call in [synchronize_data.py](./synchronize_data.py) automatically updates the dataset and store it in the local files.

Fear and Greed indices calculation are done in [fear_and_greed.py](./fear_and_greed.py)

Table schema in the database can be found at the [schema file](./schema.sql)

## Jobs
* [Indonesian bonds yield scraping (temporary)](./.github/workflows/scrape-bonds-data.yaml):
Scheduled to run every day at 1:00 AM UTC+7
* (TO-DO) Fear and Greed data update

## Libraries and Dependencies
* System dependency: id_ID.utf8 locale installed
* Python: 3.12 or compatible versions
* Libraries: see [requirements.txt](./requirements.txt)

## Running Manual
### Fear and Greed Index
* python3 main.py --timeframe {timeframe of the data} --avg_period {avg period for SMA/EMA}
    --store_db {latest N data to store into DB}
* For more info, run `python3 main.py -h`