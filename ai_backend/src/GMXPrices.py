import requests
from typing import List, Any


class GMXPrices:
    def __init__(self):
        self.url = 'https://arbitrum-api.gmxinfra.io/'

    def _get(self, path: str):
        res = requests.get(self.url + path)
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
