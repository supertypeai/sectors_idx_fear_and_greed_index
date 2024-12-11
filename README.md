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
Data fetching calls in [synchronize_data.py](./synchronize_data.py) automatically updates the dataset and store it in the local files.

Fear and Greed indices calculation are done in [fear_and_greed.py](./fear_and_greed.py)

Table schema in the database can be found at the [schema file](./schema.sql)

## Parameters
Index calculation parameters can be modified and tuned in the [parameters](./parameters) directory
* Tune the indices weighting with the [indices_weight.json](./parameters/indices_weight.json) file
* Tune the moving average method with the [average_methods.json](./parameters/average_methods.json) file \
  This can be used to modify the moving average methods used i.e. Simple Moving Average (SMA) or Exponential Moving Average (EMA)

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
* `python3 main.py --timeframe {timeframe of the data} --avg_period {avg period for SMA/EMA}
    --store_db {latest N data to store into DB}`
* For more info, run `python3 main.py -h`

### Extra Utilities
#### Fetch IHSG Data
This can be used to fetch IHSG data as target data for machine learning models
* `python3 fetch_ihsg_data.py --timeframe {timeframe of the data} --avg_period {avg period for SMA}`

#### Convert Date to ISO format
This can be used to transform columns in human-readable date format into ISO format in CSV data. Outputs the data in CSV
* `python3 convert_date_to_iso.py {input file} --columns {columns that need to be transformed} -o {output target}`
