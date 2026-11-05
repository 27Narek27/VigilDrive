import unittest
from unittest.mock import Mock, patch

import requests
from telegram_delivery import post_telegram


class DeliveryTests(unittest.TestCase):
    @patch("telegram_delivery.requests.post")
    def test_success(self, post):
        post.return_value = Mock(ok=True)
        post.return_value.json.return_value = {"ok": True}
        post_telegram("https://example.invalid", timeout=10)

    @patch("telegram_delivery.requests.post")
    def test_api_failure_is_not_success(self, post):
        post.return_value = Mock(ok=True)
        post.return_value.json.return_value = {"ok": False}
        with self.assertRaises(RuntimeError):
            post_telegram("https://example.invalid")

    @patch("telegram_delivery.requests.post")
    def test_exception_does_not_expose_token(self, post):
        post.side_effect = requests.ConnectionError("private-token-in-url")
        with self.assertRaises(RuntimeError) as error:
            post_telegram("https://example.invalid")
        self.assertNotIn("private-token", str(error.exception))


if __name__ == "__main__":
    unittest.main()
