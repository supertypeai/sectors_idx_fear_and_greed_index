import locale
import datetime

from contextlib import contextmanager


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
