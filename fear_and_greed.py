import numpy as np
import pandas as pd

from datetime import datetime
from math import ceil


def scale_to_100(value, min_val, max_val):
    epsilon = 1e-9

    if value - min_val < epsilon:
        scaled_value = 0
    else:
        scaled_value = (value - min_val) / (max_val - min_val) * 100

    return scaled_value


def calculate_moving_average(df: pd.DataFrame, avg_period: int, min_period: int = None, avg_method: str = 'sma',
                             metric: str = 'mean'):
    result: pd.Series
    # Calculate moving average based on the specified method
    match avg_method.lower():
        case 'sma':
            window = df.rolling(window=avg_period, min_periods=min_period)
        case 'ema':
            window = df.ewm(span=avg_period, min_periods=min_period, adjust=False)
        case _:
            raise AssertionError('No method found for the \'avg_method\' parameter')

    match metric.lower():
        case 'mean':
            result = window.mean()
        case 'std':
            result = window.std()
        case _:
            raise AssertionError('No method found for the \'mean\' parameter')

    return result


def normalize_data(data: pd.Series, handle_na: str = None, fill_na: float = None, scale: (int, int) = None,
                   inplace=False):
    if not inplace:
        data = data.copy()

    # # Replace infinite values with NaN
    # result.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Handle NA values
    match handle_na:
        case 'drop':
            data.dropna(inplace=True)
        case 'mean':
            mean = data.mean()
            data.fillna(mean, inplace=True)
        case 'fill':
            data.fillna(fill_na, inplace=True)
        case None:
            pass

    # Scale, scale[0] as min value, and scale[1] as max value
    if scale is not None:
        if scale[0] == 0 and scale[1] == 0:
            min_max = data.aggregate(['min', 'max'])
            scale = (min_max['min'], min_max['max'])
        elif scale[0] >= scale[1]:
            raise ValueError()
        # Perform the scaling
        data = data.apply(scale_to_100, args=(scale[0], scale[1]))

    return data


def calculate_market_momentum(daily_data: pd.DataFrame, avg_period: int, avg_method: str = 'sma') -> pd.DataFrame:
    min_momentum = -5
    max_momentum = 5

    daily_momentum_df = daily_data.copy()
    # Calculate SMA for each stock
    daily_momentum_df['sma'] = daily_momentum_df.groupby('symbol')['close'].transform(
        lambda x: calculate_moving_average(x, avg_period, 1, avg_method))

    # Calculate the percentage difference from the SMA
    # Small constant to avoid division by zero
    epsilon = 1e-9
    daily_momentum_df['momentum_sma'] = ((daily_momentum_df['close'] - daily_momentum_df['sma']) / (
            daily_momentum_df['sma'] + epsilon)) * 100

    # Remove any NaNs in case any
    daily_momentum_df = daily_momentum_df.dropna(subset=['momentum_sma'])

    # Scale the momentum to a 0-100 range for the Fear and Greed Index
    daily_momentum_df['momentum'] = daily_momentum_df['momentum_sma'].apply(
        lambda x: scale_to_100(x, min_momentum, max_momentum))

    # Calculate the Market Momentum Index on a daily basis
    daily_momentum_index = daily_momentum_df.groupby('date')['momentum'].mean().reset_index()
    return daily_momentum_index


def calculate_stock_price_strength(daily_data: pd.DataFrame, avg_period: int, avg_method: str = 'sma') -> pd.DataFrame:
    daily_mean_strength_df = daily_data.copy()

    # Calculate the daily percentage change in 'close' price for each symbol on the copy
    daily_mean_strength_df['daily_return'] = daily_mean_strength_df.groupby('symbol')['close'].pct_change()

    # Calculate SMA
    daily_mean_strength_df['strength'] = daily_mean_strength_df.groupby('symbol')['daily_return'].transform(
        lambda x: calculate_moving_average(x, avg_period, avg_method=avg_method))

    # Clean and normalize data
    daily_mean_strength_df['strength'] = normalize_data(daily_mean_strength_df['strength'],
                                                        handle_na='fill', fill_na=0, scale=(0, 0), inplace=True)

    # Group by 'date' and calculate the mean of 'strength' for each day
    daily_mean_strength_index = daily_mean_strength_df.groupby('date')['strength'].mean().reset_index()

    return daily_mean_strength_index[['date', 'strength']]


def calculate_volatility(daily_data: pd.DataFrame, avg_period: int, avg_method: str = 'sma') -> pd.DataFrame:
    df_copy_vol = daily_data.copy()
    # Calculate the daily percentage change in 'close' price for each symbol
    df_copy_vol['daily_return'] = df_copy_vol.groupby('symbol')['close'].pct_change()

    # Calculate the 7-day rolling standard deviation (volatility) of the daily returns for each symbol
    volatility_7d = df_copy_vol.groupby('symbol')['daily_return'].transform(
        lambda x: calculate_moving_average(x, avg_period, avg_method=avg_method, metric='std'))

    # Clean and normalize data
    df_copy_vol['volatility'] = normalize_data(volatility_7d, handle_na='fill', fill_na=0, scale=(0, 0), inplace=True)

    df_copy_vol = df_copy_vol.groupby('date')['volatility'].mean().reset_index()

    mean_value = \
        df_copy_vol[(df_copy_vol['volatility'] != 0) & (df_copy_vol['volatility'] != 100)][
            'volatility'].mean()
    df_copy_vol['volatility'] = df_copy_vol['volatility'].replace([0, 100], mean_value)

    return df_copy_vol


def calculate_volume_breadth(daily_data: pd.DataFrame, avg_period: int, avg_method: str = 'sma') -> pd.DataFrame:
    df_vb_copy = daily_data.copy()

    # Identify advancing and declining stocks
    # Calculate whether the stock is advancing or declining based on the 'close' price
    df_vb_copy['advancing'] = df_vb_copy.groupby('symbol')['close'].diff() > 0
    df_vb_copy['declining'] = df_vb_copy.groupby('symbol')['close'].diff() < 0

    # Calculate total volume of advancing and declining stocks for each date
    # Advancing volume
    df_vb_copy['advancing_volume'] = df_vb_copy['volume'] * df_vb_copy['advancing']
    # Declining volume
    df_vb_copy['declining_volume'] = df_vb_copy['volume'] * df_vb_copy['declining']

    # Group by date to get total advancing and declining volumes for each day
    daily_volume = df_vb_copy.groupby('date').agg({
        'advancing_volume': 'sum',
        'declining_volume': 'sum'
    }).reset_index()

    # Calculate Volume Breadth
    # Avoid division by zero by handling cases where declining volume is zero
    daily_volume['volume_breadth'] = daily_volume.apply(
        lambda row: row['advancing_volume'] / row['declining_volume'] if row['declining_volume'] != 0 else np.nan,
        axis=1
    )

    # Apply the SMA to the Volume Breadth to smooth the values
    sma_7d_vb = calculate_moving_average(daily_volume['volume_breadth'], avg_period, avg_method=avg_method)

    # Clean and normalize data
    daily_volume['volume_breadth'] = normalize_data(sma_7d_vb, handle_na='fill', fill_na=0, scale=(0, 0), inplace=True)

    # Replace 0 and 100 values with the mean
    mean_volume_breadth = daily_volume['volume_breadth'].mean()
    daily_volume['volume_breadth'] = daily_volume['volume_breadth'].apply(
        lambda x: mean_volume_breadth if x == 0 or x == 100 else x)

    return daily_volume[['date', 'volume_breadth']]


def calculate_safe_haven_demand(daily_data: pd.DataFrame, bonds_data: pd.DataFrame, avg_period: int,
                                avg_method: str = 'sma') -> pd.DataFrame:
    df_idx_shd = daily_data.copy()
    df_idx_shd = df_idx_shd.sort_values(by=['symbol', 'date'])
    df_idx_shd['stock_return'] = df_idx_shd.groupby('symbol')['close'].pct_change()
    df_idx_shd['stock_return'] = df_idx_shd['stock_return'].fillna(df_idx_shd['stock_return'].mean())

    average_daily_return = df_idx_shd.groupby('date')['stock_return'].mean().reset_index()
    average_daily_return.rename(columns={'stock_return': 'average_stock_return'}, inplace=True)

    merged_data = pd.merge(average_daily_return, bonds_data, on='date', how='inner')
    merged_data = merged_data.sort_values('date', ascending=True)

    # Calculate the SMA for stocks and bonds rate
    sma_stock = calculate_moving_average(merged_data['average_stock_return'], avg_period, 1,
                                         avg_method=avg_method, metric='mean')
    sma_rate = calculate_moving_average(merged_data['rate'], avg_period, 1,
                                        avg_method=avg_method, metric='mean')

    epsilon = 1e-9  # Small constant to avoid zero division errors
    stock_return = (merged_data['average_stock_return'] - sma_stock) / (sma_stock + epsilon)
    bonds_return = (merged_data['rate'] - sma_rate) / (sma_rate + epsilon)
    merged_data['safe_haven_index'] = (stock_return - bonds_return)

    # Scale the Safe Haven Demand Index to 0-100 for Fear and Greed context
    min_val, max_val = -10, 10  # Assume min/max range for scaling to 0-100
    merged_data['safe_haven'] = merged_data['safe_haven_index'].apply(
        lambda x: scale_to_100(x, min_val, max_val)
    )

    # Clip values to its mean to smoothen the data
    mean_value = merged_data['safe_haven'].mean()
    merged_data['safe_haven'] = merged_data['safe_haven'].apply(
        lambda x: mean_value if x <= 0 or x >= 100 else x
    )
    return merged_data[['date', 'safe_haven']]


def calculate_exchange_rate_index(rate_data: pd.DataFrame, avg_period: int, avg_method: str = 'sma') -> pd.DataFrame:
    er_df = rate_data.copy()
    # Sort the data based on date to ensure the SMA are calculated from oldest to latest
    er_df = er_df.sort_values('date', ascending=True)

    # Calculate the SMA and normalize
    sma = calculate_moving_average(er_df['rate'], avg_period, avg_method=avg_method, metric='mean')
    er_df['exchange_rate'] = normalize_data(sma, handle_na='mean', scale=(0, 0), inplace=True)

    return er_df[['date', 'exchange_rate']]


def calculate_interest_rate_index(interest_data: pd.DataFrame, timeframe: int, avg_method: str = 'sma') -> pd.DataFrame:
    interest_rate = interest_data.copy()
    interest_rate['date'] = pd.to_datetime(interest_rate['date'])
    interest_rate = interest_rate.sort_values('date', ascending=True)
    interest_rate.set_index('date', inplace=True)
    end_date = pd.Timestamp.now(tz='Asia/Bangkok')
    end_date = end_date.tz_convert(None)
    latest_date = pd.Timestamp(interest_rate.index.max())

    if latest_date.date() < end_date.date():
        # Append the last available rate for the month to the current date of end_date's month if missing
        last_rate = interest_rate['rate'].iloc[-1]
        new_row = pd.DataFrame({'rate': [last_rate]}, index=[end_date])
        interest_rate = pd.concat([interest_rate, new_row])

    # Calculate the 3-month Simple Moving Average (SMA)
    sma = calculate_moving_average(interest_rate['rate'], 3, avg_method=avg_method, metric='mean')
    # Calculate the Fear and Greed Index as the difference between the rate and the SMA
    sma_diff = interest_rate['rate'] - sma

    # Clean and normalize the data
    interest_rate['interest_rate'] = normalize_data(sma_diff, handle_na='fill', fill_na=0, scale=(0, 0), inplace=True)

    # only get the last (ceil(timeframe/30) + 1) months to resample, reducing data volume and workload
    # ceil make sure past N months are captured (including current month)
    # and the extra + 1 is to compensate for the possibly not included oldest month possible
    daily_ir_df: pd.DataFrame = interest_rate.iloc[-(ceil(timeframe / 30) + 1):]
    daily_ir_df = daily_ir_df.resample('D').ffill()
    daily_ir_df.index = daily_ir_df.index.map(datetime.date)
    return daily_ir_df[['interest_rate']].tail(n=timeframe + 1)


def calculate_buffet_indicator(mcap_data: pd.DataFrame, avg_period: int, avg_method: str = 'sma') -> pd.DataFrame:
    market_cap_df = mcap_data.copy()
    market_cap_df = market_cap_df[['date', 'idx_total_market_cap']]

    # Calculate the Simple Moving Average (SMA) of the Buffett Indicator
    sma = calculate_moving_average(market_cap_df['idx_total_market_cap'], avg_period, avg_method=avg_method,
                                   metric='mean')
    # Clean and normalize the data
    market_cap_df['buffett'] = normalize_data(sma, handle_na='mean', scale=(0, 0), inplace=True)

    # Is this really needed for the calculation? It smoothens out the data quite heavily
    # # Apply adjustments to keep values within 0-100 range and handle edge cases
    # mean_scaled_value = market_cap_df['buffett'].mean()
    # market_cap_df['buffett'] = market_cap_df['buffett'].apply(
    #     lambda x: mean_scaled_value if x <= 0 or x >= 100 else x
    # )

    return market_cap_df[['date', 'buffett']]


class FearAndGreedIndex:
    _DEFAULT_WEIGHT = {
        'momentum': .125,
        'strength': .125,
        'volatility': .125,
        'volume_breadth': .125,
        'safe_haven': .125,
        'exchange_rate': .125,
        'interest_rate': .125,
        'buffett': .125,
    }

    _DEFAULT_AVG_METHOD = {
        'momentum': 'sma',
        'strength': 'sma',
        'volatility': 'sma',
        'volume_breadth': 'sma',
        'safe_haven': 'sma',
        'exchange_rate': 'sma',
        'interest_rate': 'sma',
        'buffett': 'sma',
    }

    def __init__(self, daily_data: pd.DataFrame, mcap_data: pd.DataFrame, exchange_rate_data: pd.DataFrame,
                 interest_data: pd.DataFrame, bonds_data: pd.DataFrame):
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
        return (
                x.momentum * weight['momentum'] +
                x.strength * weight['strength'] +
                x.volatility * weight['volatility'] +
                x.volume_breadth * weight['volume_breadth'] +
                x.safe_haven * weight['safe_haven'] +
                x.exchange_rate * weight['exchange_rate'] +
                x.interest_rate * weight['interest_rate'] +
                x.buffett * weight['buffett']
        )

    def calculate_fear_and_greed_index(self, timeframe: int, avg_period: int = 7, verbose=False):
        # Calculate Market Momentum
        daily_momentum_index = calculate_market_momentum(self._daily_data, avg_period, self._avg_method['momentum'])

        # Calculate Stock Price Strength
        daily_mean_strength_index = calculate_stock_price_strength(self._daily_data, avg_period,
                                                                   self._avg_method['strength'])

        # Calculate Volatility
        volatility_index = calculate_volatility(self._daily_data, avg_period, self._avg_method['volatility'])

        # Calculate Volume Breadth
        daily_volume_index = calculate_volume_breadth(self._daily_data, avg_period, self._avg_method['volume_breadth'])

        # Calculate Safe Haven Demand
        safe_haven_index = calculate_safe_haven_demand(self._daily_data, self._bonds_data, avg_period,
                                                       self._avg_method['safe_haven'])

        # Calculate Exchange Rate Index
        exchange_rate_index = calculate_exchange_rate_index(self._exchange_rate_data, avg_period,
                                                            self._avg_method['exchange_rate'])

        # Calculate Interest Rate Index
        interest_rate_index = calculate_interest_rate_index(self._interest_data, timeframe,
                                                            self._avg_method['interest_rate'])

        # Calculate Buffet Indicator Index
        buffet_indicator_index = calculate_buffet_indicator(self._mcap_data, avg_period, self._avg_method['buffett'])

        combined_indices = daily_momentum_index.set_index('date').join([
            daily_mean_strength_index.set_index('date'),
            volatility_index.set_index('date'),
            daily_volume_index.set_index('date'),
            safe_haven_index.set_index('date'),
            exchange_rate_index.set_index('date'),
            interest_rate_index,
            buffet_indicator_index.set_index('date')
        ])

        combined_indices['fear_and_greed_index'] = combined_indices.apply(
            lambda x: self._average_indices(x, self._weight),
            axis=1
        )

        if verbose:
            print(combined_indices.to_string())

        return combined_indices
