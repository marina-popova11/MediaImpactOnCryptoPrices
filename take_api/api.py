import requests
import pandas as pd

from binance.client import Client
from datetime import datetime, timedelta

def write_to_csv(data_frame, filename) -> None:
    with open(filename, 'w', encoding='utf-8') as csvfile:
        data_frame.to_csv(csvfile, index=False)
    print("Data frame was saved is csv")

class BinanceApi:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"

    def get_price(self, symbol):
        url = self.base_url + "/ticker/price"
        params = {"symbol": symbol}
        response = requests.get(url, params=params)
        return response.json()["price"]

    def get_klines(self, symbol, interval='1d', limit=1000):
        url = self.base_url + "/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }

        response = requests.get(url, params=params)
        data = response.json()

        data_frame = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])

        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            data_frame[col] = pd.to_numeric(data_frame[col])

        data_frame['open_time'] = pd.to_datetime(data_frame['open_time'], unit='ms')
        data_frame['close_time'] = pd.to_datetime(data_frame['close_time'], unit='ms')

        return data_frame

class HistoricalData:
    def __init__(self):
        self.client = Client()

    def get_historical_data(self, symbol, years_back=3, interval=Client.KLINE_INTERVAL_1DAY):
        end_day = datetime.now()
        start_day = end_day - timedelta(days=years_back * 365)
        print(f"Data for {symbol}, period: {start_day.strftime("%d %b, %Y")} - {end_day.strftime("%d %b, %Y")}")

        klines = self.client.get_historical_klines(
            symbol,
            interval,
            start_day.strftime("%d %b, %Y"),
            end_day.strftime("%d %b, %Y")
        )

        data_frame = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])

        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            data_frame[col] = pd.to_numeric(data_frame[col])

        data_frame['open_time'] = pd.to_datetime(data_frame['open_time'], unit='ms')
        data_frame['close_time'] = pd.to_datetime(data_frame['close_time'], unit='ms')

        return data_frame

if __name__ == "__main__":
    # binance = BinanceApi()
    # price = binance.get_price("BTCUSDT")
    hist = HistoricalData()
    btc_3years = hist.get_historical_data("BTCUSDT", years_back=3)
    print(btc_3years.head())
    # print(f"Current price Bitcoin: {price}")
    #
    # df = binance.get_klines("BTCUSDT", "1d", 100)
    # print(df)
    # write_to_csv(df, "bitcoin.csv")