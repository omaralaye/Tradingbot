"""
tests/test_order_manager.py
---------------------------
Unit tests for OrderManager and _MT5Proxy.
"""

from unittest.mock import MagicMock
import pytest

from data.mt5_bridge import _MT5Proxy, TRADE_RETCODE_DONE
from execution.order_manager import OrderManager


def test_mt5_proxy_unpacks_dict_positional_args():
    mock_remote = MagicMock()
    mock_remote.order_send.return_value = MagicMock(retcode=TRADE_RETCODE_DONE, order=12345)
    mock_remote.order_check.return_value = MagicMock(retcode=0, comment="Done")

    proxy = _MT5Proxy(mock_remote)

    # Calling order_send with dict positional arg
    req = {"symbol": "EURUSDm", "volume": 0.01, "action": 1}
    res = proxy.order_send(req)

    # Should call remote order_send with unpacked kwargs
    mock_remote.order_send.assert_called_once_with(symbol="EURUSDm", volume=0.01, action=1)

    # Calling order_check with dict positional arg
    res_check = proxy.order_check(req)
    mock_remote.order_check.assert_called_once_with(symbol="EURUSDm", volume=0.01, action=1)


def test_order_manager_place_market_order():
    connector = MagicMock()
    connector.ensure_connected.return_value = True
    connector.resolve_symbol.return_value = "EURUSDm"

    mock_mt5 = MagicMock()
    tick = MagicMock(ask=1.16175, bid=1.16167)
    mock_mt5.symbol_info_tick.return_value = tick
    mock_mt5.symbol_info.return_value = MagicMock(filling_mode=3)

    order_result = MagicMock()
    order_result.retcode = TRADE_RETCODE_DONE
    order_result.order = 98765
    order_result.price = 1.16175
    order_result._asdict.return_value = {"order": 98765, "price": 1.16175}
    mock_mt5.order_send.return_value = order_result

    connector.mt5 = mock_mt5
    settings = MagicMock()

    manager = OrderManager(connector, settings)
    result = manager.place_market_order("EURUSD", "buy", 0.01, 1.15800, 1.16500)

    assert result["success"] is True
    assert result["ticket"] == 98765
    assert result["entry_price"] == 1.16175
    assert result["sl"] == 1.15800
    assert result["tp"] == 1.16500
