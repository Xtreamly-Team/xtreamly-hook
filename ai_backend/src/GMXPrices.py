import requests
import pandas as pd
from typing import List, Any


class GMXPrices:
    def __init__(self, chain: str = "arbitrum"):
        self.url = f'https://{chain}-api.gmxinfra.io/'
        self.url2 = f'https://{chain}-api.gmxinfra2.io/'

    def _get(self, path: str):
        res = requests.get(self.url + path)
        res.raise_for_status()
        return res.json()

    def _get2(self, path: str, params=None):
        res = requests.get(self.url2 + path, params=params)
        res.raise_for_status()
        return res.json()

    @staticmethod
    def _get_price(data: List[Any], symbol: str):
        recs = [
            d for d in data if d["tokenSymbol"] == symbol
        ]

        return recs.pop() if len(recs) > 0 else None

    def get_live_ticker(self, symbol: str):
        data = self._get("prices/tickers")
        return self._get_price(data, symbol)

    def get_signed_price(self, symbol: str):
        data = self._get("signed_prices/latest")["signedPrices"]
        return self._get_price(data, symbol)

    def get_candles(self, symbol: str):
        params = {
            "tokenSymbol": symbol,
            "limit": 1000,
            "period": "5m",
        }
        data = self._get2("prices/candles", params)["candles"]
        columns = ["timestamp", "open", "high", "low", "close"]

        df = pd.DataFrame(data, columns=columns)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')

        return df
