"""
Smoke tests for the trading agent.
Run: python test_agent.py
All tests are offline-safe where possible; network tests are skipped when keys are absent.
"""

import asyncio
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# ── minimal env so imports don't crash ────────────────────────────────────────
os.environ.setdefault("AGENT_DB_PATH", "/tmp/test_agent_memory.db")
os.environ.setdefault("DB_PATH", "/tmp/test_trading.db")

# ── helpers ───────────────────────────────────────────────────────────────────

def _has_key(name: str) -> bool:
    return bool(os.environ.get(name, "").strip())


# ═════════════════════════════════════════════════════════════════════════════
class TestMemory(unittest.TestCase):
    """SQLite memory layer — no network required."""

    def setUp(self):
        from agent.memory import init_memory_db
        init_memory_db()

    def test_save_and_recall(self):
        from agent.memory import recall_memory, save_insight
        ok = save_insight("test_category", "NNE tends to spike on Monday opens")
        self.assertTrue(ok)
        results = recall_memory("NNE Monday")
        self.assertGreater(len(results), 0)
        self.assertIn("NNE", results[0]["content"])

    def test_pattern_alert_dedup(self):
        from agent.memory import save_pattern_alert, was_alert_sent_recently
        save_pattern_alert("NNE", "bull_flag", 0.85, 12.34)
        self.assertTrue(was_alert_sent_recently("NNE", "bull_flag", hours=1))
        self.assertFalse(was_alert_sent_recently("NNE", "cup_handle", hours=1))

    def tearDown(self):
        import sqlite3
        conn = sqlite3.connect(os.environ["AGENT_DB_PATH"])
        conn.execute("DELETE FROM insights WHERE category='test_category'")
        conn.execute("DELETE FROM pattern_alerts WHERE symbol='NNE' AND pattern='bull_flag'")
        conn.commit()
        conn.close()


# ═════════════════════════════════════════════════════════════════════════════
class TestToolRegistry(unittest.TestCase):
    """Tool registry structure — no network required."""

    def test_all_tools_registered(self):
        from agent.tools import TOOL_REGISTRY
        expected = {
            "analyze_stock", "scan_universe", "detect_patterns",
            "fetch_news", "check_earnings", "get_market_overview",
            "save_insight", "recall_memory", "send_telegram",
        }
        missing = expected - set(TOOL_REGISTRY.keys())
        self.assertFalse(missing, f"Missing tools: {missing}")

    def test_tool_definitions_format(self):
        from agent.tools import TOOL_DEFINITIONS
        self.assertEqual(len(TOOL_DEFINITIONS), 9)
        for td in TOOL_DEFINITIONS:
            self.assertIn("name", td)
            self.assertIn("description", td)
            self.assertIn("input_schema", td)
            self.assertEqual(td["input_schema"]["type"], "object")


# ═════════════════════════════════════════════════════════════════════════════
class TestAnalyzeStock(unittest.TestCase):
    """analyze_stock tool — skipped without Twelve Data key."""

    @unittest.skipUnless(_has_key("TWELVE_DATA_KEY") or _has_key("TWELVEDATAKEY"), "No Twelve Data key")
    def test_analyze_stock_returns_dict(self):
        from agent.tools import analyze_stock
        result = analyze_stock("NNE")
        self.assertIsInstance(result, dict)
        self.assertIn("symbol", result)
        self.assertEqual(result["symbol"], "NNE")

    def test_analyze_stock_bad_symbol(self):
        """Should return an error dict, not raise."""
        from agent.tools import analyze_stock
        result = analyze_stock("XXXX_FAKE_9999")
        self.assertIsInstance(result, dict)
        # Must contain either a price key or an error key
        self.assertTrue("price" in result or "error" in result)


# ═════════════════════════════════════════════════════════════════════════════
class TestDetectPatterns(unittest.TestCase):
    """detect_patterns tool — skipped without Twelve Data key."""

    @unittest.skipUnless(_has_key("TWELVE_DATA_KEY") or _has_key("TWELVEDATAKEY"), "No Twelve Data key")
    def test_detect_patterns_structure(self):
        from agent.tools import detect_patterns
        result = detect_patterns("MARA")
        self.assertIsInstance(result, dict)
        self.assertIn("patterns", result)
        self.assertIsInstance(result["patterns"], list)


# ═════════════════════════════════════════════════════════════════════════════
class TestSendTelegram(unittest.TestCase):
    """send_telegram tool — mocked so no real messages are sent."""

    def test_send_telegram_no_token(self):
        """Without a token the function should return an error dict, not crash."""
        env_backup = os.environ.pop("TELEGRAM_TOKEN", None)
        os.environ.pop("BOT_TOKEN", None)
        try:
            from agent.tools import send_telegram
            result = send_telegram("test message")
            self.assertIsInstance(result, dict)
        finally:
            if env_backup:
                os.environ["TELEGRAM_TOKEN"] = env_backup

    @patch("httpx.post")
    def test_send_telegram_calls_api(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "result": {"message_id": 1}}
        mock_post.return_value = mock_response

        os.environ["TELEGRAM_TOKEN"] = "fake_token_for_test"
        try:
            from agent import tools as t
            # Reload to pick up the new env var
            import importlib
            importlib.reload(t)
            result = t.send_telegram("hello from test", chat_id=12345)
            self.assertTrue(mock_post.called)
            url_called = mock_post.call_args[0][0]
            self.assertIn("sendMessage", url_called)
        finally:
            os.environ.pop("TELEGRAM_TOKEN", None)


# ═════════════════════════════════════════════════════════════════════════════
class TestAgentLoop(unittest.IsolatedAsyncioTestCase):
    """Full agent loop — skipped without Anthropic API key."""

    @unittest.skipUnless(_has_key("ANTHROPIC_API_KEY"), "No Anthropic API key")
    async def test_run_agent_returns_string(self):
        from agent.core import run_agent_async
        result = await run_agent_async(
            "מה המצב הטכני של SPY? תשובה קצרה.",
            chat_id=0,   # chat_id=0 → transparency steps are no-ops (no token)
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 5)

    @unittest.skipUnless(_has_key("ANTHROPIC_API_KEY"), "No Anthropic API key")
    async def test_run_agent_uses_tool(self):
        """Ask something that forces tool use."""
        from agent.core import run_agent_async
        result = await run_agent_async(
            "כמה עולה NNE עכשיו? השתמש בכלי analyze_stock.",
            chat_id=0,
        )
        self.assertIsInstance(result, str)
        # Should contain a dollar sign or the word "שגיאה"
        self.assertTrue("$" in result or "שגיאה" in result or "NNE" in result)


# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Trading Agent — Smoke Tests")
    print("=" * 60)
    print(f"ANTHROPIC_API_KEY:  {'✅ set' if _has_key('ANTHROPIC_API_KEY') else '⬜ not set (agent loop tests skipped)'}")
    print(f"TWELVE_DATA_KEY:    {'✅ set' if _has_key('TWELVE_DATA_KEY') or _has_key('TWELVEDATAKEY') else '⬜ not set (market data tests skipped)'}")
    print(f"TELEGRAM_TOKEN:     {'✅ set' if _has_key('TELEGRAM_TOKEN') else '⬜ not set'}")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [TestMemory, TestToolRegistry, TestAnalyzeStock,
                TestDetectPatterns, TestSendTelegram, TestAgentLoop]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
