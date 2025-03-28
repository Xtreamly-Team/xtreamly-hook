import os
import requests


def get_volatility_state(symbol: str = "ETH"):
    headers = {
        "x-api-key": os.getenv("XTREAMLY_API_KEY"),
    }
    res = requests.get("https://api.xtreamly.io/state_recognize", {"symbol": symbol}, headers=headers)
    res.raise_for_status()

    return res.json()
