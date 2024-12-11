import datetime

from argparse import ArgumentParser

from synchronize_data import fetch_ihsg_data
from fear_and_greed import normalize_data, calculate_moving_average

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--timeframe",
        type=int,
        default=11,
        help="the timeframe of the data (from current date) that will be used for calculation, "
        "defaults to 11",
    )
    parser.add_argument(
        "--avg_period",
        type=int,
        default=7,
        help="the moving average period, defaults to 7",
    )

    args = parser.parse_args()
    timeframe: int = args.timeframe
    avg_period: int = args.avg_period

    # Fetch IHSG data as the target variable
    ihsg_data = fetch_ihsg_data(timeframe)

    ihsg_data = ihsg_data.drop(columns=["index_code"])
    ihsg_data["price_scaled"] = normalize_data(ihsg_data["price"], scale=(0, 0))
    ihsg_data["change"] = ihsg_data["price"].pct_change()
    ihsg_data["change_scaled"] = normalize_data(ihsg_data["change"], scale=(0, 0))
    ihsg_data["sma"] = calculate_moving_average(ihsg_data["price"], avg_period, avg_method="sma", metric="mean")
    ihsg_data["sma_scaled"] = normalize_data(ihsg_data["sma"], scale=(0,0))

    ihsg_data.to_csv(f"output/IHSG_SMA{avg_period}_{datetime.datetime.today().date()}.csv", index=False)
