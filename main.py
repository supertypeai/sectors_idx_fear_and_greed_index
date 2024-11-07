from fear_and_greed import *
from fetch_historical_data import *

if __name__ == "__main__":
    daily_data = fetch_daily_data()
    mcap_data = fetch_mcap_data()
    exchange_rate_data = fetch_idr_usd_rate()
    interest_data = fetch_idr_interest_rate()
    bonds_data = fetch_temp_bonds_rate()

    fear_and_greed_index = calculate_fear_and_greed_index(daily_data, mcap_data, exchange_rate_data, interest_data, bonds_data, True)
    print(fear_and_greed_index.to_string())
