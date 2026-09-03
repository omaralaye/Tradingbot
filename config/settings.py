"""
config/settings.py
------------------
Pydantic-based configuration loader.
All secrets and tuneable parameters come from environment variables
or a .env file — never hardcoded. See .env.example for all keys.
"""

from functools import lru_cache
from typing import Optional

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central config for the trading bot.

    Values are loaded (in priority order) from:
      1. Environment variables
      2. A .env file in the project root
      3. Defaults defined below
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- MT5 connection ---
    # On Linux, MT5 runs inside Wine. Set mt5_path to the Linux filesystem path
    # to terminal64.exe inside the Wine prefix. The connector converts it to a
    # Windows path (C:\...) automatically before passing it to MT5.
    mt5_login: int = 0
    mt5_password: SecretStr = SecretStr("")
    mt5_server: str = "MetaQuotes-Demo"
    mt5_path: str = "/home/omara/.mt5/drive_c/Program Files/MetaTrader 5/terminal64.exe"

    # --- MT5 RPyC bridge (Linux/Wine only) ---
    # The trading bot connects to the MetaTrader5 Windows API via an RPyC bridge.
    # Start the bridge before running the bot: bash scripts/start_mt5_bridge.sh
    mt5_bridge_host: str = "127.0.0.1"
    mt5_bridge_port: int = 18812
    mt5_bridge_timeout: int = 60  # RPyC sync request timeout (seconds)

    # --- Trading mode ---
    # IMPORTANT: must be explicitly set to true in .env; default is DEMO only
    enable_live_trading: bool = False

    # --- Risk parameters (must be user-set, not assumed) ---
    risk_per_trade_pct: float = 1.0       # % of account balance risked per trade
    max_daily_loss_pct: float = 3.0       # halt trading if daily loss exceeds this %
    max_open_positions: int = 5           # hard cap on concurrent open trades
    max_correlated_positions: int = 3     # max positions in correlated pairs

    # --- Signal thresholds ---
    min_signal_confidence: float = 0.60  # minimum composite confidence to take a trade
    min_timeframe_agreement: int = 2     # minimum number of TFs that must agree

    # --- Logging ---
    log_level: str = "INFO"
    log_dir: str = "logs"

    # --- Online Continuous Learning ---
    enable_online_learning: bool = True
    online_memory_path: str = "logs/online_memory.json"
    symbol_cooldown_seconds: int = 1800       # 30 min cooldown between orders on same symbol
    min_stop_loss_pips: float = 12.0          # min SL floor to avoid tight chop stop-outs
    quarantine_loss_streak: int = 2           # consecutive losses to trigger setup quarantine
    quarantine_duration_seconds: int = 14400  # 4 hours quarantine

    # --- Optional alerting ---
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance.

    Using lru_cache ensures the .env file is parsed only once per process.
    Call get_settings.cache_clear() in tests to reset between test cases.
    """
    return Settings()
