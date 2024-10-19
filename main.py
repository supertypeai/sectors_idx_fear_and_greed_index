from fear_and_greed import *
from fetch_historical_data import *

if __name__ == "__main__":
    daily_data = fetch_daily_data()
    idr_usd_data = fetch_idr_usd_rate()
    print(idr_usd_data)
    interest_date = fetch_idr_interest_rate()
    print(interest_date)
    fear_and_greed_index = calculate_fear_and_greed_index(daily_data)
    print(fear_and_greed_index)
