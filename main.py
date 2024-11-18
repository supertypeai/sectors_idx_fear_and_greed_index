import json
from argparse import ArgumentParser
from fear_and_greed import *
from synchronize_data import *

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--timeframe', type=int, default=11,
                        help='the timeframe of the data (from current date) that will be used for calculation, '
                             'defaults to 11')
    parser.add_argument('--avg_period', type=int, default=7,
                        help='the moving average period, defaults to 7')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='prints the final indices results')
    parser.add_argument('--store_db', type=int, default=0,
                        help='specifies how many latest datapoints are to be updated into database')

    args = parser.parse_args()
    timeframe: int = args.timeframe
    avg_period: int = args.avg_period
    verbose: bool = args.verbose
    store_db: int = args.store_db

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
                                                          timeframe, avg_period, weight, verbose)

    if store_db > 0:
        push_to_db(fear_and_greed_index, n_latest=store_db)
