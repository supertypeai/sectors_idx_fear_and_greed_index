import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.preprocessing import MinMaxScaler


def scale_to_100(value, min_val, max_val):
    epsilon = 1e-9

    if max_val - min_val < epsilon:
        scaled_value = 0
    else:
        scaled_value = (value - min_val) / (max_val - min_val) * 100

    return np.clip(scaled_value, 0, 100)


def calculate_market_momentum(daily_data: pd.DataFrame):
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
    daily_momentum_df['momentum_scaled'] = daily_momentum_df['momentum_sma'].apply(
        lambda x: scale_to_100(x, min_momentum, max_momentum))

    # Calculate the Market Momentum Index on a daily basis
    daily_momentum_index = daily_momentum_df.groupby('date')['momentum_scaled'].mean().reset_index()
    return daily_momentum_index


def calculate_stock_price_strength(daily_data: pd.DataFrame):
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
    daily_mean_strength_df['scaled_strength_index'] = daily_mean_strength_df['sma_7d_return'].apply(
        lambda x: scale_to_100(x, min_sma, max_sma))

    daily_mean_strength_df = daily_mean_strength_df[['symbol', 'date', 'scaled_strength_index']].dropna()
    # Group by 'date' and calculate the mean of 'scaled_strength_index' for each day
    daily_mean_strength_index = daily_mean_strength_df.groupby('date')['scaled_strength_index'].mean().reset_index()

    return daily_mean_strength_index


def calculate_volatility(daily_data: pd.DataFrame):
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
    df_copy_vol['scaled_volatility_index'] = df_copy_vol['volatility_7d'].apply(
        lambda x: scale_to_100(x, min_vol, max_vol))

    df_copy_vol = df_copy_vol[['symbol', 'date', 'scaled_volatility_index']].dropna()
    df_copy_vol = df_copy_vol.groupby('date')['scaled_volatility_index'].mean().reset_index()

    mean_value = \
    df_copy_vol[(df_copy_vol['scaled_volatility_index'] != 0) & (df_copy_vol['scaled_volatility_index'] != 100)][
        'scaled_volatility_index'].mean()
    df_copy_vol['scaled_volatility_index'] = df_copy_vol['scaled_volatility_index'].replace([0, 100], mean_value)

    return df_copy_vol


def calculate_volume_breadth(daily_data: pd.DataFrame):
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
    daily_volume['scaled_vb'] = daily_volume['sma_7d_vb'].apply(lambda x: scale_to_100(x, min_vb, max_vb))

    # Replace 0 and 100 values with the mean
    mean_scaled_vb = daily_volume['scaled_vb'].mean()
    daily_volume['scaled_vb'] = daily_volume['scaled_vb'].apply(lambda x: mean_scaled_vb if x == 0 or x == 100 else x)

    return daily_volume


def calculate_safe_haven_demand(daily_data: pd.DataFrame, bonds_data: pd.DataFrame):
    df_idx_shd = daily_data.copy()
    df_idx_shd = df_idx_shd.sort_values(by=['symbol', 'date'])
    df_idx_shd['stock_return'] = df_idx_shd.groupby('symbol')['close'].pct_change() * 100
    df_idx_shd.loc[:, 'stock_return'].fillna(df_idx_shd['stock_return'].mean(), inplace=True)

    average_daily_return = df_idx_shd.groupby('date')['stock_return'].mean().reset_index()
    average_daily_return.rename(columns={'stock_return': 'average_stock_return'}, inplace=True)

    merged_data = pd.merge(average_daily_return, bonds_data, on='date', how='inner')

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

    min_val, max_val = -5, 5
    merged_data['safe_haven_scaled'] = merged_data['safe_haven_index'].apply(
        lambda x: scale_to_100(x, min_val, max_val)
    )
    mean_value = merged_data['safe_haven_scaled'].mean()
    merged_data['safe_haven_scaled'] = merged_data['safe_haven_scaled'].apply(
        lambda x: mean_value if x < 0 or x > 100 else x
    )
    return merged_data


def calculate_exchange_rate_index(rate_data: pd.DataFrame):
    rate_data['SMA_7'] = rate_data['rate'].rolling(window=7).mean()
    rate_data['SMA_7'].fillna(rate_data['SMA_7'].mean(), inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 100))
    rate_data['scaled_fear_greed'] = scaler.fit_transform(rate_data[['SMA_7']])

    return rate_data


def calculate_interest_rate_index(interest_data: pd.DataFrame):
    interest_data.set_index('date', inplace=True)
    today = pd.Timestamp(datetime.now().date())
    end_date = pd.Timestamp(today)
    latest_date = pd.Timestamp(interest_data.index.max())

    if latest_date.to_period('M') < end_date.to_period('M'):
        # Append the last available rate for the month starting on the 1st of end_date's month if missing
        last_rate = interest_data['rate'].iloc[-1]
        new_row = pd.DataFrame({'rate': [last_rate]}, index=[end_date.replace(day=1)])
        interest_rate = pd.concat([interest_data, new_row])

    daily_ir_df = interest_rate.resample('D').ffill()
    # Calculate the 7-day Simple Moving Average (SMA)
    daily_ir_df['SMA_7'] = daily_ir_df['rate'].rolling(window=7).mean()
    # Calculate the Fear and Greed Index as the difference between the rate and the SMA
    daily_ir_df['fear_greed_index'] = daily_ir_df['rate'] - daily_ir_df['SMA_7']
    # Handle initial NaN values in SMA calculation (first 6 days) by filling them with 0 for now
    daily_ir_df['fear_greed_index'].fillna(0, inplace=True)
    # Scale the Fear and Greed Index to a range between 0 and 100
    scaler = MinMaxScaler(feature_range=(0, 100))
    # Reshape for scaling to avoid any zero division issue and fit_transform safely
    daily_ir_df['scaled_fear_greed'] = scaler.fit_transform(daily_ir_df[['fear_greed_index']])
    # Handle out-of-bound values (replace ≤0 and ≥100 values with the mean)
    mean_value = daily_ir_df['scaled_fear_greed'].mean()

    # Apply constraints: Replace values ≤ 0 or ≥ 100 with the mean value
    daily_ir_df['scaled_fear_greed'] = daily_ir_df['scaled_fear_greed'].apply(
        lambda x: mean_value if x <= 0 or x >= 100 else x
    )
    return daily_ir_df[['date', 'scaled_fear_greed']]


def calculate_buffet_indicator(daily_data: pd.DataFrame):
    # TODO fix data used in this calculation
    market_cap_df = daily_data.copy()
    market_cap_df = market_cap_df[['date', 'market_cap']]


def calculate_fear_and_greed_index(daily_data: pd.DataFrame, mcap_data: pd.DataFrame, idr_usd_data: pd.DataFrame,
                                   bi_interest_data: pd.DataFrame):
    # Calculate Market Momentum
    daily_momentum_index = calculate_market_momentum(daily_data)
    print(daily_momentum_index)

    # Calculate Stock Price Strength
    daily_mean_strength_index = calculate_stock_price_strength(daily_data)
    print(daily_mean_strength_index)

    # Calculate Volatility
    volatility_index = calculate_volatility(daily_data)
    print(volatility_index)

    # Calculate Volume Breadth
    daily_volume_index = calculate_volume_breadth(daily_data)
    print(daily_volume_index)
    pass
