import numpy as np
import pandas as pd

from datetime import datetime
from math import ceil
from sklearn.preprocessing import MinMaxScaler


def scale_to_100(value, min_val, max_val):
    epsilon = 1e-9

    if max_val - min_val < epsilon:
        scaled_value = 0
    else:
        scaled_value = (value - min_val) / (max_val - min_val) * 100

    return np.clip(scaled_value, 0, 100)


def calculate_market_momentum(daily_data: pd.DataFrame) -> pd.DataFrame:
    sma_period = 7
    min_momentum = -5
    max_momentum = 5

    daily_momentum_df = daily_data.copy()
    # Calculate SMA for each stock
    daily_momentum_df['sma'] = daily_momentum_df.groupby('symbol')['close'].transform(
        lambda x: x.rolling(window=sma_period, min_periods=1).mean())

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


def calculate_stock_price_strength(daily_data: pd.DataFrame) -> pd.DataFrame:
    daily_mean_strength_df = daily_data.copy()

    # Calculate the daily percentage change in 'close' price for each symbol on the copy
    daily_mean_strength_df['daily_return'] = daily_mean_strength_df.groupby('symbol')['close'].pct_change() * 100

    # Calculate the 7-day SMA of the daily returns on the copy
    daily_mean_strength_df['sma_7d_return'] = daily_mean_strength_df.groupby('symbol')['daily_return'].transform(
        lambda x: x.rolling(window=7).mean())

    # Handle zero division by replacing any infinite or NaN values in 'sma_7d_return' with zeros
    daily_mean_strength_df['sma_7d_return'] = daily_mean_strength_df['sma_7d_return'].replace([np.inf, -np.inf], np.nan)
    daily_mean_strength_df['sma_7d_return'] = daily_mean_strength_df['sma_7d_return'].fillna(0)

    # Scale the SMA results for fear and greed index on a 0 to 100 scale
    # Calculate the min and max of the SMA to normalize
    min_sma = daily_mean_strength_df['sma_7d_return'].min()
    max_sma = daily_mean_strength_df['sma_7d_return'].max()
    daily_mean_strength_df['strength'] = daily_mean_strength_df['sma_7d_return'].apply(
        lambda x: scale_to_100(x, min_sma, max_sma))

    daily_mean_strength_df = daily_mean_strength_df[['symbol', 'date', 'strength']].dropna()
    # Group by 'date' and calculate the mean of 'strength' for each day
    daily_mean_strength_index = daily_mean_strength_df.groupby('date')['strength'].mean().reset_index()

    return daily_mean_strength_index


def calculate_volatility(daily_data: pd.DataFrame) -> pd.DataFrame:
    df_copy_vol = daily_data.copy()
    # Calculate the daily percentage change in 'close' price for each symbol
    df_copy_vol['daily_return'] = df_copy_vol.groupby('symbol')['close'].pct_change()

    # Calculate the 7-day rolling standard deviation (volatility) of the daily returns for each symbol
    df_copy_vol['volatility_7d'] = df_copy_vol.groupby('symbol')['daily_return'].transform(
        lambda x: x.rolling(window=7).std())

    # Handle zero division by replacing any infinite or NaN values in 'volatility_7d' with zeros
    df_copy_vol['volatility_7d'] = df_copy_vol['volatility_7d'].fillna(0)
    df_copy_vol['volatility_7d'] = df_copy_vol['volatility_7d'].replace([np.inf, -np.inf], np.nan)

    # Scale the volatility on a 0 to 100 scale for the fear and greed index context
    min_vol = df_copy_vol['volatility_7d'].min()
    max_vol = df_copy_vol['volatility_7d'].max()
    df_copy_vol['volatility'] = df_copy_vol['volatility_7d'].apply(
        lambda x: scale_to_100(x, min_vol, max_vol))

    df_copy_vol = df_copy_vol[['symbol', 'date', 'volatility']].dropna()
    df_copy_vol = df_copy_vol.groupby('date')['volatility'].mean().reset_index()

    mean_value = \
        df_copy_vol[(df_copy_vol['volatility'] != 0) & (df_copy_vol['volatility'] != 100)][
            'volatility'].mean()
    df_copy_vol['volatility'] = df_copy_vol['volatility'].replace([0, 100], mean_value)

    return df_copy_vol


def calculate_volume_breadth(daily_data: pd.DataFrame) -> pd.DataFrame:
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

    # Apply the 7-day SMA to the Volume Breadth to smooth the values
    daily_volume['sma_7d_vb'] = daily_volume['volume_breadth'].rolling(window=7).mean()

    # Handle NaN or infinite values in the SMA with zeros (for stability)
    daily_volume['sma_7d_vb'] = daily_volume['sma_7d_vb'].replace([np.inf, -np.inf], np.nan)
    daily_volume['sma_7d_vb'] = daily_volume['sma_7d_vb'].fillna(0)

    # Scale the Volume Breadth for the fear and greed index context
    # Min and max for scaling
    min_vb = daily_volume['sma_7d_vb'].min()
    max_vb = daily_volume['sma_7d_vb'].max()
    daily_volume['volume_breadth'] = daily_volume['sma_7d_vb'].apply(lambda x: scale_to_100(x, min_vb, max_vb))

    # Replace 0 and 100 values with the mean
    mean_volume_breadth = daily_volume['volume_breadth'].mean()
    daily_volume['volume_breadth'] = daily_volume['volume_breadth'].apply(lambda x: mean_volume_breadth if x == 0 or x == 100 else x)

    return daily_volume[['date', 'volume_breadth']]


def calculate_safe_haven_demand(daily_data: pd.DataFrame, bonds_data: pd.DataFrame) -> pd.DataFrame:
    df_idx_shd = daily_data.copy()
    df_idx_shd = df_idx_shd.sort_values(by=['symbol', 'date'])
    df_idx_shd['stock_return'] = df_idx_shd.groupby('symbol')['close'].pct_change() * 100
    df_idx_shd['stock_return'] = df_idx_shd['stock_return'].fillna(df_idx_shd['stock_return'].mean())

    average_daily_return = df_idx_shd.groupby('date')['stock_return'].mean().reset_index()
    average_daily_return.rename(columns={'stock_return': 'average_stock_return'}, inplace=True)

    merged_data = pd.merge(average_daily_return, bonds_data, on='date', how='inner')
    merged_data = merged_data.sort_values('date', ascending=True)

    sma_period = 7
    merged_data['sma_stock'] = merged_data['average_stock_return'].rolling(window=sma_period, min_periods=1).mean()
    merged_data['sma_rate'] = merged_data['rate'].rolling(window=sma_period, min_periods=1).mean()

    epsilon = 1e-9  # Small constant to avoid zero division errors
    merged_data['safe_haven_index'] = (
                                              (merged_data['average_stock_return'] - merged_data['sma_stock']) / (
                                              merged_data['sma_stock'] + epsilon) -
                                              (merged_data['rate'] - merged_data['sma_rate']) / (
                                                      merged_data['sma_rate'] + epsilon)
                                      ) * 100

    # Scale the Safe Haven Demand Index to 0-100 for Fear and Greed context
    min_val, max_val = -1000, 1000  # Assume min/max range for scaling to 0-100
    merged_data['safe_haven'] = merged_data['safe_haven_index'].apply(
        lambda x: scale_to_100(x, min_val, max_val)
    )
    mean_value = merged_data['safe_haven'].mean()
    merged_data['safe_haven'] = merged_data['safe_haven'].apply(
        lambda x: mean_value if x <= 0 or x >= 100 else x
    )
    return merged_data[['date', 'safe_haven']]


def calculate_exchange_rate_index(rate_data: pd.DataFrame) -> pd.DataFrame:
    er_df = rate_data.copy()
    er_df = er_df.sort_values('date', ascending=True)
    er_df['SMA_7'] = er_df['rate'].rolling(window=7).mean()
    er_df['SMA_7'] = er_df['SMA_7'].fillna(er_df['SMA_7'].mean())
    scaler = MinMaxScaler(feature_range=(0, 100))
    er_df['exchange_rate'] = scaler.fit_transform(er_df[['SMA_7']])

    return er_df[['date', 'exchange_rate']]


def calculate_interest_rate_index(interest_data: pd.DataFrame, timeframe: int) -> pd.DataFrame:
    interest_rate = interest_data.copy()
    interest_rate['date'] = pd.to_datetime(interest_rate['date'])
    interest_rate = interest_rate.sort_values('date', ascending=True)
    interest_rate.set_index('date', inplace=True)
    end_date = datetime.now()
    latest_date = pd.Timestamp(interest_rate.index.max())

    if latest_date.month < end_date.month:
        # Append the last available rate for the month to the current date of end_date's month if missing
        last_rate = interest_rate['rate'].iloc[-1]
        new_row = pd.DataFrame({'rate': [last_rate]}, index=[end_date])
        interest_rate = pd.concat([interest_rate, new_row])

    # Calculate the 3-month Simple Moving Average (SMA)
    interest_rate['SMA_3'] = interest_rate['rate'].rolling(window=3).mean()
    # Calculate the Fear and Greed Index as the difference between the rate and the SMA
    interest_rate['fear_greed_index'] = interest_rate['rate'] - interest_rate['SMA_3']
    # Handle initial NaN values in SMA calculation by filling them with 0 for now
    interest_rate['fear_greed_index'] = interest_rate['fear_greed_index'].fillna(0)
    # Scale the Fear and Greed Index to a range between 0 and 100
    scaler = MinMaxScaler(feature_range=(0, 100))
    # Reshape for scaling to avoid any zero division issue and fit_transform safely
    interest_rate['interest_rate'] = scaler.fit_transform(interest_rate[['fear_greed_index']])

    # # Handle out-of-bound values (replace ≤0 and ≥100 values with the mean)
    # mean_value = interest_rate['interest_rate'].mean()
    # # Apply constraints: Replace values ≤ 0 or ≥ 100 with the mean value
    # interest_rate['interest_rate'] = interest_rate['interest_rate'].apply(
    #     lambda x: mean_value if x <= 0 or x >= 100 else x
    # )

    # only get the last (ceil(timeframe/30) + 1) months to resample, reducing data volume and workload
    # ceil make sure past N months are captured (including current month)
    # and the extra + 1 is to compensate for the possibly not included oldest month possible
    daily_ir_df: pd.DataFrame = interest_rate.iloc[-(ceil(timeframe/30) + 1):]
    daily_ir_df = daily_ir_df.resample('D').ffill()
    daily_ir_df.index = daily_ir_df.index.map(datetime.date)
    return daily_ir_df[['interest_rate']].tail(n=timeframe+1)


def calculate_buffet_indicator(mcap_data: pd.DataFrame) -> pd.DataFrame:
    market_cap_df = mcap_data.copy()
    market_cap_df = market_cap_df[['date', 'idx_total_market_cap']]

    # Indonesian GDP for 2024 in Indonesian Rupiah (IDR)
    # Using an approximate exchange rate: 1 USD = 15,000 IDR
    usd_to_idr_exchange_rate = 15000
    indonesia_gdp_idr = 1.47 * 10 ** 12 * usd_to_idr_exchange_rate  # 1.47 trillion USD to IDR

    market_cap_df['buffett_indicator'] = (market_cap_df['idx_total_market_cap'] / indonesia_gdp_idr) * 100

    # Drop rows with NaN values in the 'buffett_indicator' column
    market_cap_df = market_cap_df.dropna(subset=['buffett_indicator'])

    # Calculate the 7-day Simple Moving Average (SMA) of the Buffett Indicator
    market_cap_df['buffett_SMA_7'] = market_cap_df['buffett_indicator'].rolling(window=7).mean()
    # Fill any missing values (NaNs) in the SMA with the mean of the SMA column
    market_cap_df['buffett_SMA_7'] = market_cap_df['buffett_SMA_7'].fillna(market_cap_df['buffett_SMA_7'].mean())

    # Scale the Buffett Indicator to a 0-100 range using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 100))
    # Scale by fitting and transforming on reshaped data
    market_cap_df['buffett'] = scaler.fit_transform(market_cap_df[['buffett_SMA_7']])
    # Step 4: Apply adjustments to keep values within 0-100 range and handle edge cases
    mean_scaled_value = market_cap_df['buffett'].mean()
    market_cap_df['buffett'] = market_cap_df['buffett'].apply(
        lambda x: mean_scaled_value if x <= 0 or x >= 100 else x
    )

    return market_cap_df[['date', 'buffett']]


def average_indices(x: pd.Series, weight=None) -> float:
    if weight is None:
        weight = {
            'momentum': .125,
            'strength': .125,
            'volatility': .125,
            'volume_breadth': .125,
            'safe_haven': .125,
            'exchange_rate': .125,
            'interest_rate': .125,
            'buffett': .125,
        }
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


def calculate_fear_and_greed_index(daily_data: pd.DataFrame, mcap_data: pd.DataFrame, exchange_rate_data: pd.DataFrame,
                                   interest_data: pd.DataFrame, bonds_data: pd.DataFrame,
                                   timeframe: int, weight: dict[str, float], verbose=False):
    # Calculate Market Momentum
    daily_momentum_index = calculate_market_momentum(daily_data)

    # Calculate Stock Price Strength
    daily_mean_strength_index = calculate_stock_price_strength(daily_data)

    # Calculate Volatility
    volatility_index = calculate_volatility(daily_data)

    # Calculate Volume Breadth
    daily_volume_index = calculate_volume_breadth(daily_data)

    # Calculate Safe Haven Demand
    safe_haven_index = calculate_safe_haven_demand(daily_data, bonds_data)

    # Calculate Exchange Rate Index
    exchange_rate_index = calculate_exchange_rate_index(exchange_rate_data)

    # Calculate Interest Rate Index
    interest_rate_index = calculate_interest_rate_index(interest_data, timeframe)

    # Calculate Buffet Indicator Index
    buffet_indicator_index = calculate_buffet_indicator(mcap_data)

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
        lambda x: average_indices(x, weight),
        axis=1
    )

    if verbose:
        print(combined_indices.to_string())

    return combined_indices
