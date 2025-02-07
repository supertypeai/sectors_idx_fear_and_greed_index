import numpy as np
import pandas as pd

from datetime import datetime
from math import ceil
from utils import calculate_moving_average, normalize_data, DIV_EPSILON


def calculate_market_momentum(
    daily_data: pd.DataFrame, avg_period: int, avg_method: str = "sma"
) -> pd.DataFrame:
    daily_momentum_df = daily_data.copy()
    # Calculate SMA for each stock
    daily_momentum_df["sma"] = daily_momentum_df.groupby("symbol")["close"].transform(
        lambda x: calculate_moving_average(x, avg_period, 1, avg_method)
    )

    # Calculate the percentage difference from the SMA
    daily_momentum_sma_chg = (daily_momentum_df["close"] - daily_momentum_df["sma"]) / (
        daily_momentum_df["sma"] + DIV_EPSILON
    )

    # Normalize and scale data
    daily_momentum_df["momentum"] = normalize_data(
        daily_momentum_sma_chg, handle_na="drop", scale=(0, 0)
    )

    # Calculate the Market Momentum Index on a daily basis
    daily_momentum_index = (
        daily_momentum_df.groupby("date")["momentum"].mean().reset_index()
    )
    return daily_momentum_index


def calculate_stock_price_strength(
    daily_data: pd.DataFrame, avg_period: int, avg_method: str = "sma"
) -> pd.DataFrame:
    daily_mean_strength_df = daily_data.copy()

    # Calculate the daily percentage change in 'close' price for each symbol on the copy
    daily_mean_strength_df["daily_return"] = daily_mean_strength_df.groupby("symbol")[
        "close"
    ].pct_change()

    # Calculate SMA
    daily_mean_strength_df["strength"] = daily_mean_strength_df.groupby("symbol")[
        "daily_return"
    ].transform(
        lambda x: calculate_moving_average(x, avg_period, avg_method=avg_method)
    )

    # Clean and normalize data
    daily_mean_strength_df["strength"] = normalize_data(
        daily_mean_strength_df["strength"],
        handle_na="fill",
        fill_na=0,
        scale=(0, 0),
        inplace=True,
    )

    # Group by 'date' and calculate the mean of 'strength' for each day
    daily_mean_strength_index = (
        daily_mean_strength_df.groupby("date")["strength"].mean().reset_index()
    )

    return daily_mean_strength_index[["date", "strength"]]


def calculate_volatility(
    daily_data: pd.DataFrame, avg_period: int, avg_method: str = "sma"
) -> pd.DataFrame:
    df_copy_vol = daily_data.copy()
    # Calculate the daily percentage change in 'close' price for each symbol
    df_copy_vol["daily_return"] = df_copy_vol.groupby("symbol")["close"].pct_change()

    # Calculate the 7-day rolling standard deviation (volatility) of the daily returns for each symbol
    def calculate_sma_and_clean(x):
        volatility_7d = calculate_moving_average(
            x, avg_period, avg_method=avg_method, metric="std"
        )
        return normalize_data(
            volatility_7d, handle_na="fill", fill_na=0, scale=(0, 0), inplace=True
        )

    df_copy_vol["volatility"] = df_copy_vol.groupby("symbol")["daily_return"].transform(
        lambda x: calculate_sma_and_clean(x)
    )

    df_copy_vol = df_copy_vol.groupby("date")["volatility"].mean().reset_index()

    mean_value = df_copy_vol[
        (df_copy_vol["volatility"] != 0) & (df_copy_vol["volatility"] != 100)
    ]["volatility"].mean()
    df_copy_vol["volatility"] = df_copy_vol["volatility"].replace([0, 100], mean_value)

    return df_copy_vol


def calculate_volume_breadth(
    daily_data: pd.DataFrame, avg_period: int, avg_method: str = "sma"
) -> pd.DataFrame:
    df_vb_copy = daily_data.copy()

    # Identify advancing and declining stocks
    # Calculate whether the stock is advancing or declining based on the 'close' price
    df_vb_copy["advancing"] = df_vb_copy.groupby("symbol")["close"].diff() > 0
    df_vb_copy["declining"] = df_vb_copy.groupby("symbol")["close"].diff() < 0

    # Calculate total volume of advancing and declining stocks for each date
    # Advancing volume
    df_vb_copy["advancing_volume"] = df_vb_copy["volume"] * df_vb_copy["advancing"]
    # Declining volume
    df_vb_copy["declining_volume"] = df_vb_copy["volume"] * df_vb_copy["declining"]

    # Group by date to get total advancing and declining volumes for each day
    daily_volume = (
        df_vb_copy.groupby("date")
        .agg({"advancing_volume": "sum", "declining_volume": "sum"})
        .reset_index()
    )

    # Calculate Volume Breadth
    # Avoid division by zero by handling cases where declining volume is zero
    daily_volume["vb_ratio"] = daily_volume.apply(
        lambda row: (
            row["advancing_volume"] / row["declining_volume"]
            if row["declining_volume"] != 0 and row["advancing_volume"] != 0
            else DIV_EPSILON
        ),
        axis=1,
    )

    # log scale the ratio to smoothen data for linear scaling
    daily_volume["vb_log_scaled"] = np.log(daily_volume["vb_ratio"])

    # Apply the SMA to the Volume Breadth to smooth the values
    daily_volume["sma_7d_vb"] = calculate_moving_average(
        daily_volume["vb_log_scaled"], avg_period, avg_method=avg_method
    )

    # Clean and normalize data
    daily_volume["volume_breadth"] = normalize_data(
        daily_volume["sma_7d_vb"],
        handle_na="fill",
        fill_na=0,
        scale=(0, 0),
        inplace=True,
    )

    return daily_volume[["date", "volume_breadth"]]


def calculate_safe_haven_demand(
    daily_data: pd.DataFrame,
    bonds_data: pd.DataFrame,
    avg_period: int,
    avg_method: str = "sma",
) -> pd.DataFrame:
    df_idx_shd = daily_data.copy()
    df_idx_shd = df_idx_shd.sort_values(by=["symbol", "date"])
    df_idx_shd["stock_return"] = df_idx_shd.groupby("symbol")["close"].pct_change()
    df_idx_shd["stock_return"] = df_idx_shd["stock_return"].fillna(
        df_idx_shd["stock_return"].mean()
    )

    average_daily_return = (
        df_idx_shd.groupby("date")["stock_return"].mean().reset_index()
    )
    average_daily_return.rename(
        columns={"stock_return": "average_stock_return"}, inplace=True
    )

    merged_data = pd.merge(average_daily_return, bonds_data, on="date", how="inner")
    merged_data = merged_data.sort_values("date", ascending=True)

    # Calculate the SMA for stocks and bonds rate
    sma_stock = calculate_moving_average(
        merged_data["average_stock_return"],
        avg_period,
        1,
        avg_method=avg_method,
        metric="mean",
    )
    sma_rate = calculate_moving_average(
        merged_data["rate"], avg_period, 1, avg_method=avg_method, metric="mean"
    )

    stock_return = (merged_data["average_stock_return"] - sma_stock) / (
        sma_stock + DIV_EPSILON
    )
    bonds_return = (merged_data["rate"] - sma_rate) / (sma_rate + DIV_EPSILON)
    safe_haven_index = stock_return - bonds_return

    # Scale the Safe Haven Demand Index to 0-100 for Fear and Greed context
    merged_data["safe_haven"] = normalize_data(safe_haven_index, scale=(0, 0))

    # # Clip values to its mean to smoothen the data
    # mean_value = merged_data['safe_haven'].mean()
    # merged_data['safe_haven'] = merged_data['safe_haven'].apply(
    #     lambda x: mean_value if x <= 0 or x >= 100 else x
    # )
    return merged_data[["date", "safe_haven"]]


def calculate_exchange_rate_index(
    rate_data: pd.DataFrame, avg_period: int, avg_method: str = "sma"
) -> pd.DataFrame:
    er_df = rate_data.copy()
    # Sort the data based on date to ensure the SMA are calculated from oldest to latest
    er_df = er_df.sort_values("date", ascending=True)

    # Calculate the SMA and normalize
    sma = calculate_moving_average(
        er_df["rate"], avg_period, avg_method=avg_method, metric="mean"
    )
    er_df["exchange_rate"] = normalize_data(
        sma, handle_na="mean", scale=(0, 0), inplace=True
    )

    return er_df[["date", "exchange_rate"]]


def calculate_interest_rate_index(
    interest_data: pd.DataFrame, timeframe: int, avg_method: str = "sma"
) -> pd.DataFrame:
    interest_rate = interest_data.copy()
    interest_rate["date"] = pd.to_datetime(interest_rate["date"])
    interest_rate = interest_rate.sort_values("date", ascending=True)
    interest_rate.set_index("date", inplace=True)
    end_date = pd.Timestamp.now(tz="Asia/Bangkok")
    end_date = end_date.tz_convert(None)
    latest_date = pd.Timestamp(interest_rate.index.max())

    if latest_date.date() < end_date.date():
        # Append the last available rate for the month to the current date of end_date's month if missing
        last_rate = interest_rate["rate"].iloc[-1]
        new_row = pd.DataFrame({"rate": [last_rate]}, index=[end_date])
        interest_rate = pd.concat([interest_rate, new_row])

    # Calculate the 3-month Simple Moving Average (SMA)
    sma = calculate_moving_average(
        interest_rate["rate"], 3, avg_method=avg_method, metric="mean"
    )
    # Calculate the Fear and Greed Index as the difference between the rate and the SMA
    sma_diff = interest_rate["rate"] - sma

    # Clean and normalize the data
    interest_rate["interest_rate"] = normalize_data(
        sma_diff, handle_na="fill", fill_na=0, scale=(0, 0), inplace=True
    )

    # only get the last (ceil(timeframe/30) + 1) months to resample, reducing data volume and workload
    # ceil make sure past N months are captured (including current month)
    # Add an extra +1 to account if the current date is not the date a new data point is inserted
    # and another extra + 1 is to compensate for the possibly not included oldest month possible
    daily_ir_df: pd.DataFrame = interest_rate.iloc[-(ceil(timeframe / 30) + 2) :]
    daily_ir_df = daily_ir_df.resample("D").ffill()
    daily_ir_df.index = daily_ir_df.index.map(datetime.date)
    return daily_ir_df[["interest_rate"]].tail(n=timeframe + 1)


def calculate_buffet_indicator(
    mcap_data: pd.DataFrame, avg_period: int, avg_method: str = "sma"
) -> pd.DataFrame:
    market_cap_df = mcap_data.copy()
    market_cap_df = market_cap_df[["date", "idx_total_market_cap"]]

    # Calculate the Simple Moving Average (SMA) of the Buffett Indicator
    sma = calculate_moving_average(
        market_cap_df["idx_total_market_cap"],
        avg_period,
        avg_method=avg_method,
        metric="mean",
    )
    # Clean and normalize the data
    market_cap_df["buffett"] = normalize_data(
        sma, handle_na="mean", scale=(0, 0), inplace=True
    )

    # Is this really needed for the calculation? It smoothens out the data quite heavily
    # # Apply adjustments to keep values within 0-100 range and handle edge cases
    # mean_scaled_value = market_cap_df['buffett'].mean()
    # market_cap_df['buffett'] = market_cap_df['buffett'].apply(
    #     lambda x: mean_scaled_value if x <= 0 or x >= 100 else x
    # )

    return market_cap_df[["date", "buffett"]]


class FearAndGreedIndex:
    _DEFAULT_WEIGHT = {
        "momentum": 0.125,
        "strength": 0.125,
        "volatility": 0.125,
        "volume_breadth": 0.125,
        "safe_haven": 0.125,
        "exchange_rate": 0.125,
        "interest_rate": 0.125,
        "buffett": 0.125,
    }

    _DEFAULT_AVG_METHOD = {
        "momentum": "sma",
        "strength": "sma",
        "volatility": "sma",
        "volume_breadth": "sma",
        "safe_haven": "sma",
        "exchange_rate": "sma",
        "interest_rate": "sma",
        "buffett": "sma",
    }

    _FEATURES = [
        "momentum",
        "strength",
        "volatility",
        "volume_breadth",
        "safe_haven",
        "exchange_rate",
        "interest_rate",
        "buffett",
    ]

    def __init__(
        self,
        daily_data: pd.DataFrame,
        mcap_data: pd.DataFrame,
        exchange_rate_data: pd.DataFrame,
        interest_data: pd.DataFrame,
        bonds_data: pd.DataFrame,
    ):
        # Indices data
        self._daily_data = daily_data
        self._mcap_data = mcap_data
        self._exchange_rate_data = exchange_rate_data
        self._interest_data = interest_data
        self._bonds_data = bonds_data

        # Auxiliary parameters
        self._avg_method: dict[str, str] = self._DEFAULT_AVG_METHOD
        self._weight: dict[str, float] = self._DEFAULT_WEIGHT

    def set_weight(self, weight: dict[str, float]):
        self._weight = weight

    def set_moving_avg_method(self, moving_avg_method: dict[str, str]):
        self._avg_method = moving_avg_method

    @staticmethod
    def _average_indices(x: pd.Series, weight=None) -> float:
        if weight is None:
            weight = FearAndGreedIndex._DEFAULT_WEIGHT
        return sum(
            x[feature] * weight[feature] for feature in FearAndGreedIndex._FEATURES
        )

    def calculate_fear_and_greed_index(
        self, timeframe: int, avg_period: int = 7, verbose=False
    ):
        # Calculate Market Momentum
        daily_momentum_index = calculate_market_momentum(
            self._daily_data, avg_period, self._avg_method["momentum"]
        )

        # Calculate Stock Price Strength
        daily_mean_strength_index = calculate_stock_price_strength(
            self._daily_data, avg_period, self._avg_method["strength"]
        )

        # Calculate Volatility
        volatility_index = calculate_volatility(
            self._daily_data, avg_period, self._avg_method["volatility"]
        )

        # Calculate Volume Breadth
        daily_volume_index = calculate_volume_breadth(
            self._daily_data, avg_period, self._avg_method["volume_breadth"]
        )

        # Calculate Safe Haven Demand
        safe_haven_index = calculate_safe_haven_demand(
            self._daily_data,
            self._bonds_data,
            avg_period,
            self._avg_method["safe_haven"],
        )

        # Calculate Exchange Rate Index
        exchange_rate_index = calculate_exchange_rate_index(
            self._exchange_rate_data, avg_period, self._avg_method["exchange_rate"]
        )

        # Calculate Interest Rate Index
        interest_rate_index = calculate_interest_rate_index(
            self._interest_data, timeframe, self._avg_method["interest_rate"]
        )

        # Calculate Buffet Indicator Index
        buffet_indicator_index = calculate_buffet_indicator(
            self._mcap_data, avg_period, self._avg_method["buffett"]
        )

        combined_indices = daily_momentum_index.set_index("date").join(
            [
                daily_mean_strength_index.set_index("date"),
                volatility_index.set_index("date"),
                daily_volume_index.set_index("date"),
                safe_haven_index.set_index("date"),
                exchange_rate_index.set_index("date"),
                interest_rate_index,
                buffet_indicator_index.set_index("date"),
            ]
        )

        combined_indices["fear_and_greed_index"] = combined_indices.apply(
            lambda x: self._average_indices(x, self._weight), axis=1
        )

        if verbose:
            print(combined_indices.to_string())

        return combined_indices
