import locale
import datetime
import pandas as pd

from contextlib import contextmanager

DIV_EPSILON = 1e-9  # Small constant to avoid zero division errors


@contextmanager
def override_locale(category, locale_string):
    prev_locale_string = locale.getlocale(category)
    locale.setlocale(category, locale_string)
    yield
    locale.setlocale(category, prev_locale_string)


def id_date_format_to_datetime(date: str):
    parsed_datetime = datetime.datetime.strptime(date, '%d %B %Y')
    return parsed_datetime.date()


def get_previous_workday(date: datetime.date):
    date_weekday = date.weekday()
    if date_weekday == 0:  # monday
        day_delta = 3
    elif date_weekday == 6:  # sunday
        day_delta = 2
    else:
        day_delta = 1
    return date - datetime.timedelta(day_delta)


def scale_to_100(value, min_val, max_val):
    if value - min_val < DIV_EPSILON:
        scaled_value = 0
    else:
        scaled_value = (value - min_val) / (max_val - min_val) * 100

    return scaled_value


def calculate_moving_average(
    df: pd.DataFrame,
    avg_period: int,
    min_period: int = None,
    avg_method: str = "sma",
    metric: str = "mean",
):
    result: pd.Series
    # Calculate moving average based on the specified method
    match avg_method.lower():
        case "sma":
            window = df.rolling(window=avg_period, min_periods=min_period)
        case "ema":
            window = df.ewm(span=avg_period, min_periods=min_period, adjust=False)
        case _:
            raise AssertionError("No method found for the 'avg_method' parameter")

    match metric.lower():
        case "mean":
            result = window.mean()
        case "std":
            result = window.std()
        case _:
            raise AssertionError("No method found for the 'mean' parameter")

    return result


def normalize_data(
    data: pd.Series,
    handle_na: str = None,
    fill_na: float = None,
    scale: (int, int) = None,
    inplace=False,
):
    if not inplace:
        data = data.copy()

    # # Replace infinite values with NaN
    # result.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Handle NA values
    match handle_na:
        case "drop":
            data.dropna(inplace=True)
        case "mean":
            mean = data.mean()
            data.fillna(mean, inplace=True)
        case "fill":
            data.fillna(fill_na, inplace=True)
        case None:
            pass

    # Scale, scale[0] as min value, and scale[1] as max value
    if scale is not None:
        if scale[0] == 0 and scale[1] == 0:
            min_max = data.aggregate(["min", "max"])
            scale = (min_max["min"], min_max["max"])
        elif scale[0] >= scale[1]:
            raise ValueError()
        # Perform the scaling
        data = data.apply(scale_to_100, args=(scale[0], scale[1]))

    return data

