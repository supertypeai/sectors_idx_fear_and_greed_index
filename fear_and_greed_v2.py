import numpy as np
import pandas as pd

from datetime import datetime
from utils import calculate_moving_average, normalize_data, DIV_EPSILON


def _calculate_momentum(
        daily_data: pd.DataFrame, avg_period: int, avg_method: str = "sma"
) -> pd.DataFrame:
    df_momentum = daily_data.copy()

    df_momentum["rolling_mean"] = calculate_moving_average(df_momentum["price"], avg_period, avg_method=avg_method)
    df_momentum["_momentum"] = ((df_momentum['price'] - df_momentum['rolling_mean']) / df_momentum['rolling_mean'])
    df_momentum["momentum"] = normalize_data(df_momentum["_momentum"], scale=(0, 0))

    return df_momentum[["date", "momentum"]]


def _calculate_strength(
        daily_data: pd.DataFrame, avg_period: int, avg_method: str = "sma"
) -> pd.DataFrame:
    df_strength = daily_data.copy()

    df_strength["daily_change"] = df_strength["price"].diff()
    df_strength["up_days"] = (df_strength["daily_change"] > 0).astype(int)
    df_strength["_strength"] = calculate_moving_average(df_strength["up_days"], avg_period, avg_method=avg_method)
    df_strength["strength"] = normalize_data(df_strength["_strength"], scale=(0, 0))

    return df_strength[["date", "strength"]]


def _calculate_volatility(
        daily_data: pd.DataFrame, avg_period: int, avg_method: str = "sma"
) -> pd.DataFrame:
    df_volatility = daily_data.copy()

    df_volatility["daily_return"] = df_volatility["price"].pct_change()
    df_volatility["_volatility"] = calculate_moving_average(df_volatility["daily_return"], avg_period,
                                                           avg_method=avg_method, metric="std")
    df_volatility["volatility"] = normalize_data(df_volatility["_volatility"], scale=(0, 0))

    return df_volatility[["date", "volatility"]]


def _calculate_recovery(
        daily_data: pd.DataFrame, rolling_window: int
) -> pd.DataFrame:
    df_recovery = daily_data.copy()
    df_recovery['high'] = df_recovery['price'].rolling(window=rolling_window).max()
    df_recovery['low'] = df_recovery['price'].rolling(window=rolling_window).min()
    df_recovery['_recovery'] = ((df_recovery['price'] - df_recovery['low']) /
                               (df_recovery['high'] - df_recovery['low']))
    df_recovery["recovery"] = normalize_data(df_recovery["_recovery"], scale=(0, 0))

    return df_recovery[["date", "recovery"]]


def _calculate_trend_strength(
        daily_data: pd.DataFrame,
        trend_window: int
) -> pd.DataFrame:
    df_trend_strength = daily_data.copy()
    df_trend_strength["_trend_strength"] = df_trend_strength["price"].pct_change(periods=trend_window)
    df_trend_strength["trend_strength"] = normalize_data(df_trend_strength["_trend_strength"], scale=(0, 0))

    return df_trend_strength[["date", "trend_strength"]]


def shift_data(daily_data: pd.DataFrame, period: int):
    df_shifted_data = daily_data.copy()
    df_shifted_data["change"] = df_shifted_data["price"].pct_change()
    df_shifted_data["change"] = normalize_data(df_shifted_data["change"], scale=(0, 0))

    df_shifted_data["date"] = (pd.to_datetime(df_shifted_data["date"]) - pd.tseries.offsets.BusinessDay(n=period)).dt.date
    df_shifted_data.set_index("date", inplace=True)
    return df_shifted_data


class FearAndGreedIndexV2:
    _DEFAULT_WEIGHT = {
        "momentum": 0.2,
        "strength": 0.2,
        "volatility": 0.2,
        "recovery": 0.2,
        "trend_strength": 0.2,
        "_intercept": 0,
    }

    _DEFAULT_AVG_METHOD = {
        "momentum": "sma",
        "strength": "sma",
        "volatility": "sma",
        # "recovery": "sma",
        # "trend_strength": "sma",
    }

    _FEATURES = [
        "momentum",
        "strength",
        "volatility",
        "recovery",
        "trend_strength",
    ]

    def __init__(
            self,
            daily_data: pd.DataFrame,
    ):
        # Indices data
        self._daily_data = daily_data

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
            weight = FearAndGreedIndexV2._DEFAULT_WEIGHT
        return np.clip(
            sum(
                x[feature] * weight[feature] for feature in FearAndGreedIndexV2._FEATURES
            ) + weight["_intercept"],
            0,
            100
        )

    def calculate_fear_and_greed_index(
            self, correlate: int | None = None, verbose=False
    ):
        # Calculate Momentum
        df_momentum = _calculate_momentum(
            self._daily_data, 90, self._avg_method["momentum"]
        )
        # Calculate Strength
        df_strength = _calculate_strength(
            self._daily_data, 23, self._avg_method["strength"]
        )
        # Calculate Volatility
        df_volatility = _calculate_volatility(
            self._daily_data, 52, self._avg_method["volatility"]
        )
        # Calculate Recovery
        df_recovery = _calculate_recovery(
            self._daily_data, 49
        )
        # Calculate Trend Strength
        df_trend_strength = _calculate_trend_strength(
            self._daily_data, 9
        )

        combined_indices = df_momentum.set_index("date").join(
            [
                df_strength.set_index("date"),
                df_volatility.set_index("date"),
                df_recovery.set_index("date"),
                df_trend_strength.set_index("date"),
            ]
        )

        combined_indices["fear_and_greed_index"] = combined_indices.apply(
            lambda x: self._average_indices(x, self._weight), axis=1
        )

        if correlate is not None:
            lagged_ihsg = shift_data(self._daily_data, 3)
            if verbose:
                print(f"{correlate}-days Lagged IHSG data")
                print(lagged_ihsg.tail(10).to_string())
            joined_data = combined_indices.join(lagged_ihsg)
            corr = joined_data["fear_and_greed_index"].corr(joined_data["change"])
            combined_indices[f"corr_{correlate}d"] = corr

        if verbose:
            print("Final result")
            print(combined_indices.tail(10).to_string())

        return combined_indices
