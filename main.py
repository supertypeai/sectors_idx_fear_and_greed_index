from argparse import ArgumentParser
from fear_and_greed import *
from fetch_historical_data import *

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--timeframe', type=int, default=11)

    args = parser.parse_args()
    timeframe = args.timeframe

    daily_data = fetch_daily_data(timeframe)
    mcap_data = fetch_mcap_data(timeframe)
    exchange_rate_data = fetch_idr_usd_rate(timeframe)
    interest_data = fetch_idr_interest_rate()
    bonds_data = fetch_temp_bonds_rate(timeframe)

    fear_and_greed_index = calculate_fear_and_greed_index(daily_data, mcap_data,
                                                          exchange_rate_data, interest_data,
                                                          bonds_data,
                                                          timeframe, False)
    print(fear_and_greed_index.to_string())
