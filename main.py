from fear_and_greed import *

if __name__ == "__main__":
    daily_data = fetch_daily_data()
    fear_and_greed_index = calculate_fear_and_greed_index(daily_data)
    print(fear_and_greed_index)
