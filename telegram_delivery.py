"""Validate Telegram responses without exposing token-bearing URLs in errors."""
import requests


def post_telegram(url, **kwargs):
    try:
        response = requests.post(url, **kwargs)
        if not response.ok or response.json().get("ok") is not True:
            raise RuntimeError("Telegram rejected the message")
    except (requests.RequestException, ValueError):
        raise RuntimeError("Telegram delivery failed; check connectivity and bot settings") from None
