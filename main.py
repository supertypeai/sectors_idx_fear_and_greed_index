from fear_and_greed import *
from fetch_historical_data import *

if __name__ == "__main__":
    daily_data = fetch_daily_data()
    mcap_data = fetch_mcap_data()
    idr_usd_data = fetch_idr_usd_rate()
    bi_interest_data = fetch_idr_interest_rate()

    fear_and_greed_index = calculate_fear_and_greed_index(daily_data, mcap_data, idr_usd_data, bi_interest_data)
    print(fear_and_greed_index)
