import json
from argparse import ArgumentParser
from fear_and_greed_v2 import FearAndGreedIndexV2
from synchronize_data import (
    fetch_ihsg_data,
    push_to_db,
)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--timeframe",
        type=int,
        default=180,
        help="the timeframe of the data (from current date) that will be used for calculation, "
        "defaults to 180",
    )
    parser.add_argument(
        "--correlate",
        type=int,
        nargs="?",
        help="specifies the forward shift of the IHSG data for correlation calculation",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="prints the final indices results"
    )
    parser.add_argument(
        "--store-db",
        type=int,
        default=0,
        help="specifies how many latest datapoints are to be updated into database",
    )

    args = parser.parse_args()
    timeframe: int = args.timeframe
    correlate_period: int = args.correlate
    verbose: bool = args.verbose
    store_db: int = args.store_db

    daily_data = fetch_ihsg_data(timeframe)

    with open("parameters/v2/indices_weight.json") as f:
        weight = json.load(f)

    with open("parameters/v2/average_methods.json") as f:
        moving_avg_methods = json.load(f)

    fear_and_greed_index = FearAndGreedIndexV2(daily_data)
    fear_and_greed_index.set_weight(weight)
    fear_and_greed_index.set_moving_avg_method(moving_avg_methods)
    fear_and_greed_data = fear_and_greed_index.calculate_fear_and_greed_index(correlate_period, verbose)

    if store_db > 0:
        push_to_db(fear_and_greed_data, n_latest=store_db)
