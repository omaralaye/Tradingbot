"""
execution/order_manager.py
---------------------------
Places, modifies, and closes orders via the MT5 Python API.

Every order attempt and result is logged. Market orders retry on
requote errors (up to 3 times). All MT5 error codes are mapped to
human-readable messages.
"""

from __future__ import annotations

import time
from typing import Optional

from loguru import logger

from data.mt5_connector import MT5Connector
from data.mt5_bridge import (
    ORDER_TYPE_BUY, ORDER_TYPE_SELL,
    ORDER_TYPE_BUY_LIMIT, ORDER_TYPE_SELL_LIMIT,
    TRADE_ACTION_DEAL, TRADE_ACTION_PENDING,
    TRADE_ACTION_SLTP, TRADE_ACTION_MODIFY, TRADE_ACTION_REMOVE,
    ORDER_FILLING_IOC, ORDER_FILLING_FOK, ORDER_TIME_GTC,
    TRADE_RETCODE_DONE, TRADE_RETCODE_REQUOTE,
)


# Human-readable MT5 return code descriptions
MT5_RETCODE_MESSAGES = {
    10004: "Requote",
    10006: "Request rejected",
    10007: "Request cancelled by trader",
    10008: "Order placed",
    10009: "Request completed",
    10010: "Only part of the request was completed",
    10011: "Request processing error",
    10012: "Request cancelled by timeout",
    10013: "Invalid request",
    10014: "Invalid trade volume",
    10015: "Invalid price",
    10016: "Invalid stops",
    10017: "Trade is disabled",
    10018: "Market is closed",
    10019: "Insufficient funds",
    10020: "Prices changed",
    10021: "No quotes to process the request",
    10022: "Invalid order expiration date",
    10023: "Order state changed",
    10024: "Too many requests",
    10025: "No changes in request",
    10026: "Autotrading disabled by server",
    10027: "Autotrading disabled by client terminal",
    10028: "Request locked for processing",
    10029: "Order or position frozen",
    10030: "Invalid order filling type",
}


class OrderManager:
    """Manages all trade execution via MT5.

    Separates execution concerns from signal/risk logic.
    All methods log every attempt, result, and error code.
    """

    MAX_REQUOTE_RETRIES = 3

    def __init__(self, connector: MT5Connector, settings):
        """
        Args:
            connector: Active MT5Connector instance.
            settings:  Settings (used to check ENABLE_LIVE_TRADING flag).
        """
        self._connector = connector
        self._settings  = settings

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def place_market_order(
        self,
        symbol:     str,
        order_type: str,   # 'buy' or 'sell'
        lot_size:   float,
        sl_price:   float,
        tp_price:   float,
        comment:    str = "TradingBot",
    ) -> dict:
        """Place a market order with SL and TP.

        Args:
            symbol:     Instrument symbol.
            order_type: 'buy' or 'sell'.
            lot_size:   Volume in lots.
            sl_price:   Stop loss price.
            tp_price:   Take profit price.
            comment:    Order comment (visible in MT5 trade history).

        Returns:
            Dict with 'success' (bool), 'ticket' (int or None), 'error' (str or None).
        """
        if not self._connector.ensure_connected():
            return self._error_result("MT5 not connected.")

        mt5 = self._connector.mt5
        broker_symbol = self._connector.resolve_symbol(symbol) or symbol
        mt5_type = ORDER_TYPE_BUY if order_type == "buy" else ORDER_TYPE_SELL

        tick = mt5.symbol_info_tick(broker_symbol)
        if tick is None:
            return self._error_result(f"Failed to get tick price for {broker_symbol}")
        price = tick.ask if order_type == "buy" else tick.bid

        info = mt5.symbol_info(broker_symbol)
        filling_type = ORDER_FILLING_IOC
        if info is not None and getattr(info, "filling_mode", None) is not None:
            if info.filling_mode == 1:
                filling_type = ORDER_FILLING_FOK
            elif info.filling_mode == 2:
                filling_type = ORDER_FILLING_IOC

        request = {
            "action":       TRADE_ACTION_DEAL,
            "symbol":       broker_symbol,
            "volume":       lot_size,
            "type":         mt5_type,
            "price":        price,
            "sl":           sl_price,
            "tp":           tp_price,
            "deviation":    20,       # allowed slippage in points
            "magic":        20240101,   # unique magic number for this bot
            "comment":      comment,
            "type_time":    ORDER_TIME_GTC,
            "type_filling": filling_type,
        }

        logger.info(
            "Placing {} market order | {} (broker: {}) | lot={} | price={:.5f} | SL={:.5f} | TP={:.5f}",
            order_type.upper(), symbol, broker_symbol, lot_size, price, sl_price, tp_price,
        )

        for attempt in range(1, self.MAX_REQUOTE_RETRIES + 1):
            result = mt5.order_send(request)
            if result is None:
                return self._error_result(f"order_send returned None. Error: {mt5.last_error()}")

            if result.retcode == TRADE_RETCODE_DONE:
                logger.info(
                    "Order placed successfully | ticket={} | {}",
                    result.order, broker_symbol,
                )
                return {
                    "success": True,
                    "ticket": result.order,
                    "entry_price": getattr(result, "price", price),
                    "sl": sl_price,
                    "tp": tp_price,
                    "error": None,
                    "result": result._asdict() if hasattr(result, "_asdict") else {},
                }

            if result.retcode == TRADE_RETCODE_REQUOTE:
                logger.warning("Requote on attempt {}/{}. Retrying...", attempt, self.MAX_REQUOTE_RETRIES)
                time.sleep(0.5 * attempt)
                # Update tick price on requote
                tick = mt5.symbol_info_tick(broker_symbol)
                if tick is not None:
                    request["price"] = tick.ask if order_type == "buy" else tick.bid
                continue

            # Non-retryable error
            msg = self._handle_order_error(result.retcode)
            logger.error("Order failed | {} | retcode={} | {}", broker_symbol, result.retcode, msg)
            return self._error_result(msg)

        return self._error_result("Max requote retries exceeded.")

    def place_limit_order(
        self,
        symbol:       str,
        order_type:   str,
        lot_size:     float,
        entry_price:  float,
        sl_price:     float,
        tp_price:     float,
    ) -> dict:
        """Place a pending limit order."""
        if not self._connector.ensure_connected():
            return self._error_result("MT5 not connected.")

        mt5 = self._connector.mt5
        broker_symbol = self._connector.resolve_symbol(symbol) or symbol
        mt5_type = (
            ORDER_TYPE_BUY_LIMIT if order_type == "buy"
            else ORDER_TYPE_SELL_LIMIT
        )

        info = mt5.symbol_info(broker_symbol)
        filling_type = ORDER_FILLING_IOC
        if info is not None and getattr(info, "filling_mode", None) is not None:
            if info.filling_mode == 1:
                filling_type = ORDER_FILLING_FOK
            elif info.filling_mode == 2:
                filling_type = ORDER_FILLING_IOC

        request = {
            "action":       TRADE_ACTION_PENDING,
            "symbol":       broker_symbol,
            "volume":       lot_size,
            "type":         mt5_type,
            "price":        entry_price,
            "sl":           sl_price,
            "tp":           tp_price,
            "magic":        20240101,
            "type_time":    ORDER_TIME_GTC,
            "type_filling": filling_type,
        }
        result = mt5.order_send(request)
        if result and result.retcode == TRADE_RETCODE_DONE:
            logger.info("Limit order placed | ticket={} | {}", result.order, broker_symbol)
            return {
                "success": True,
                "ticket": result.order,
                "entry_price": entry_price,
                "sl": sl_price,
                "tp": tp_price,
                "error": None,
            }
        msg = self._handle_order_error(result.retcode if result else -1)
        return self._error_result(msg)

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def modify_sl_tp(self, ticket: int, new_sl: float, new_tp: float) -> bool:
        """Modify the SL and TP of an open position."""
        if not self._connector.ensure_connected():
            logger.error("MT5 not connected — cannot modify SL/TP.")
            return False
        mt5 = self._connector.mt5
        request = {
            "action":   TRADE_ACTION_SLTP,
            "position": ticket,
            "sl":       new_sl,
            "tp":       new_tp,
        }
        result = mt5.order_send(request)
        ok = result is not None and result.retcode == TRADE_RETCODE_DONE
        if ok:
            logger.info("Modified SL/TP for ticket {}", ticket)
        else:
            logger.error("Failed to modify ticket {}: {}", ticket, result.retcode if result else "None")
        return ok

    def close_position(self, ticket: int) -> bool:
        """Close a single open position by ticket number."""
        if not self._connector.ensure_connected():
            logger.error("MT5 not connected — cannot close position.")
            return False
        mt5 = self._connector.mt5
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.warning("Position {} not found.", ticket)
            return False

        pos    = positions[0]
        symbol = pos.symbol
        volume = pos.volume
        close_type = ORDER_TYPE_SELL if pos.type == ORDER_TYPE_BUY else ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error("Failed to get tick price for symbol {}", symbol)
            return False
        price = tick.bid if close_type == ORDER_TYPE_SELL else tick.ask

        info = mt5.symbol_info(symbol)
        filling_type = ORDER_FILLING_IOC
        if info is not None and getattr(info, "filling_mode", None) is not None:
            if info.filling_mode == 1:
                filling_type = ORDER_FILLING_FOK
            elif info.filling_mode == 2:
                filling_type = ORDER_FILLING_IOC

        request = {
            "action":       TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       volume,
            "type":         close_type,
            "position":     ticket,
            "price":        price,
            "deviation":    20,
            "magic":        20240101,
            "comment":      "TradingBot close",
            "type_time":    ORDER_TIME_GTC,
            "type_filling": filling_type,
        }
        result = mt5.order_send(request)
        ok = result is not None and result.retcode == TRADE_RETCODE_DONE
        if ok:
            logger.info("Position {} closed.", ticket)
        else:
            logger.error("Failed to close position {}: {}", ticket, result.retcode if result else "None")
        return ok

    def close_all_positions(self) -> list[dict]:
        """Close all open positions. Called by kill switch callback."""
        if not self._connector.ensure_connected():
            logger.error("MT5 not connected — cannot close all positions.")
            return []
        mt5 = self._connector.mt5
        positions = mt5.positions_get()
        results   = []
        if not positions:
            logger.info("No open positions to close.")
            return results

        logger.warning("Closing all {} open positions (kill switch).", len(positions))
        for pos in positions:
            ok = self.close_position(pos.ticket)
            results.append({"ticket": pos.ticket, "symbol": pos.symbol, "closed": ok})
        return results

    def get_open_positions(self) -> list[dict]:
        """Return a list of all currently open positions as dicts."""
        if not self._connector.is_connected():
            return []
        try:
            mt5 = self._connector.mt5
            positions = mt5.positions_get()
            if not positions:
                return []
            return [p._asdict() for p in positions]
        except Exception as exc:
            logger.error("Failed to get open positions: {}", exc)
            return []

    def get_account_info(self) -> dict:
        """Return current account balance, equity, and free margin."""
        if not self._connector.is_connected():
            return {"balance": 0.0, "equity": 0.0, "margin_free": 0.0}
        try:
            mt5 = self._connector.mt5
            info = mt5.account_info()
            if info is None:
                return {"balance": 0.0, "equity": 0.0, "margin_free": 0.0}
            return {
                "balance":     info.balance,
                "equity":      info.equity,
                "margin_free": info.margin_free,
                "currency":    info.currency,
            }
        except Exception as exc:
            logger.error("Failed to get account info: {}", exc)
            return {"balance": 0.0, "equity": 0.0, "margin_free": 0.0}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _handle_order_error(self, retcode: int) -> str:
        """Map MT5 return code to a human-readable error message."""
        return MT5_RETCODE_MESSAGES.get(retcode, f"Unknown MT5 error code: {retcode}")

    @staticmethod
    def _error_result(message: str) -> dict:
        logger.error("Order error: {}", message)
        return {"success": False, "ticket": None, "error": message}
