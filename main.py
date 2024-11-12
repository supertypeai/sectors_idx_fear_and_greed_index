import json
from argparse import ArgumentParser
from fear_and_greed import *
from synchronize_data import *

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--timeframe', type=int, default=11)
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    timeframe: int = args.timeframe
    verbose: bool = args.verbose

    daily_data = fetch_daily_data(timeframe)
    mcap_data = fetch_mcap_data(timeframe)
    exchange_rate_data = fetch_idr_usd_rate(timeframe)
    interest_data = fetch_idr_interest_rate()
    bonds_data = fetch_temp_bonds_rate(timeframe)

    with open('indices_weight.json') as f:
        weight = json.load(f)

    fear_and_greed_index = calculate_fear_and_greed_index(daily_data, mcap_data,
                                                          exchange_rate_data, interest_data,
                                                          bonds_data,
                                                          timeframe, weight, verbose)

    push_to_db(fear_and_greed_index, n_latest=30)
