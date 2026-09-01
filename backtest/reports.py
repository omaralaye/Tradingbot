"""
backtest/reports.py
--------------------
Generates performance reports from backtest results.

Metrics reported: win rate, profit factor, max drawdown, Sharpe ratio,
Sortino ratio, and per-regime / per-session breakdowns.

A strategy should be questioned, not celebrated, if metrics look
"too good" — always check for data leakage first.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


class BacktestReporter:
    """Computes and formats backtest performance statistics."""

    def __init__(self, results: pd.DataFrame, initial_balance: float):
        """
        Args:
            results:         DataFrame of trades from BacktestEngine.get_results().
            initial_balance: Starting balance used in the backtest.
        """
        self._df      = results.copy() if not results.empty else pd.DataFrame()
        self._initial = initial_balance

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Compute aggregate performance metrics.

        Returns:
            Dict with win_rate, profit_factor, max_drawdown_pct,
            sharpe_ratio, sortino_ratio, total_trades, net_pnl.
        """
        if self._df.empty:
            return {"total_trades": 0, "note": "No trades in backtest period."}

        df = self._df.copy()
        total  = len(df)
        wins   = (df["outcome"] == "win").sum()
        losses = (df["outcome"] == "loss").sum()

        win_rate = wins / total if total > 0 else 0.0

        pnl          = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)
        gross_profit = pnl[pnl > 0].sum()
        gross_loss   = abs(pnl[pnl < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        net_pnl       = float(pnl.sum())

        # Max drawdown from equity curve
        equity = self.equity_curve()
        rolling_max  = equity.cummax()
        drawdown     = (equity - rolling_max) / rolling_max.replace(0, np.nan)
        max_drawdown = float(drawdown.min()) * 100  # as negative %

        # Sharpe ratio (annualised, assuming H1 bars)
        returns   = pnl / self._initial
        sharpe    = self._sharpe(returns, periods_per_year=8760)
        sortino   = self._sortino(returns, periods_per_year=8760)

        return {
            "total_trades":    total,
            "wins":            int(wins),
            "losses":          int(losses),
            "win_rate":        round(win_rate, 4),
            "profit_factor":   round(profit_factor, 4),
            "max_drawdown_pct": round(max_drawdown, 2),
            "sharpe_ratio":    round(sharpe, 3),
            "sortino_ratio":   round(sortino, 3),
            "net_pnl":         round(net_pnl, 2),
            "initial_balance": self._initial,
            "final_balance":   round(self._initial + net_pnl, 2),
            "total_return_pct": round(net_pnl / self._initial * 100, 2),
        }

    def per_regime_breakdown(self) -> pd.DataFrame:
        """Return win rate and profit factor grouped by market regime."""
        if self._df.empty or "regime" not in self._df.columns:
            return pd.DataFrame()
        return self._grouped_stats("regime")

    def per_session_breakdown(self) -> pd.DataFrame:
        """Return win rate and profit factor grouped by trading session."""
        if self._df.empty or "session" not in self._df.columns:
            return pd.DataFrame()
        return self._grouped_stats("session")

    def equity_curve(self) -> pd.Series:
        """Return the cumulative equity curve (starting from initial_balance)."""
        if self._df.empty:
            return pd.Series([self._initial])
        pnl = pd.to_numeric(self._df["pnl"], errors="coerce").fillna(0.0)
        return self._initial + pnl.cumsum()

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def print_report(self) -> None:
        """Print a formatted performance summary to the console."""
        stats = self.summary()
        print("\n" + "=" * 50)
        print("  BACKTEST PERFORMANCE REPORT")
        print("=" * 50)
        for key, val in stats.items():
            print(f"  {key:<22}: {val}")
        print("=" * 50 + "\n")

    def save_report(self, path: str) -> None:
        """Save summary and per-regime breakdown to JSON and CSV files.

        Args:
            path: Base path (without extension). Creates <path>_summary.json
                  and <path>_trades.csv.
        """
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)

        summary = self.summary()
        (base.parent / f"{base.stem}_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )

        if not self._df.empty:
            self._df.to_csv(str(base.parent / f"{base.stem}_trades.csv"), index=False)

        logger.info("Backtest report saved to {}", base.parent)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _grouped_stats(self, group_col: str) -> pd.DataFrame:
        """Compute win rate and profit factor per group."""
        rows = []
        for group_val, group_df in self._df.groupby(group_col):
            total    = len(group_df)
            wins     = (group_df["outcome"] == "win").sum()
            pnl      = pd.to_numeric(group_df["pnl"], errors="coerce").fillna(0.0)
            gp       = pnl[pnl > 0].sum()
            gl       = abs(pnl[pnl < 0].sum())
            pf       = gp / gl if gl > 0 else 0.0
            rows.append({
                group_col:       group_val,
                "total_trades":  total,
                "win_rate":      round(wins / total, 4) if total > 0 else 0.0,
                "profit_factor": round(pf, 4),
                "net_pnl":       round(float(pnl.sum()), 2),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
        mean = returns.mean()
        std  = returns.std()
        if std == 0 or np.isnan(std):
            return 0.0
        return float(mean / std * np.sqrt(periods_per_year))

    @staticmethod
    def _sortino(returns: pd.Series, periods_per_year: int = 252) -> float:
        mean     = returns.mean()
        downside = returns[returns < 0].std()
        if downside == 0 or np.isnan(downside):
            return 0.0
        return float(mean / downside * np.sqrt(periods_per_year))
