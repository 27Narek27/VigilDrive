import os
import unittest
from unittest.mock import patch

from settings import get_bot_token


class SettingsTests(unittest.TestCase):
    def test_missing_token_has_actionable_error(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": ""}):
            with self.assertRaisesRegex(RuntimeError, "TELEGRAM_BOT_TOKEN"):
                get_bot_token()

    def test_environment_token_is_used(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": " test-value "}):
            self.assertEqual(get_bot_token(), "test-value")
