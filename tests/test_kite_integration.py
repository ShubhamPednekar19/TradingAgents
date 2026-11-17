import asyncio

import pytest

from tradingagents.integrations import alerts, kite


def test_yahoo_symbol_maps_indian_exchanges():
    assert kite.yahoo_symbol("reliance", "NSE") == "RELIANCE.NS"
    assert kite.yahoo_symbol("TCS", "BSE") == "TCS.BO"
    assert kite.yahoo_symbol("NIFTY25NOVFUT", "NFO") == "NIFTY25NOVFUT"
    assert kite.yahoo_symbol("HFCL-BE", "NSE") == "HFCL.NS"


def test_parse_payload_unwraps_pages():
    assert kite.parse_payload('{"data": [1, 2], "pagination": {}}') == [1, 2]
    assert kite.parse_payload('{"equity": {"net": 5}}') == {"equity": {"net": 5}}
    with pytest.raises(kite.KiteError):
        kite.parse_payload("please log in")


def test_portfolio_from_kite():
    holdings = [
        {"tradingsymbol": "INFY", "exchange": "NSE", "quantity": 10, "t1_quantity": 2, "average_price": 1500.0},
        {"tradingsymbol": "SOLD", "exchange": "NSE", "quantity": 0, "t1_quantity": 0, "average_price": 1.0},
    ]
    positions = [{"tradingsymbol": "SBIN", "exchange": "BSE", "quantity": -5, "average_price": 800.0, "view": "net"},
                 {"tradingsymbol": "SBIN", "exchange": "BSE", "quantity": -5, "average_price": 800.0, "view": "day"}]
    # A delivery sale of a holding today: holdings already show it, so the -3 must not count again.
    positions.append({"tradingsymbol": "KANOHAR", "exchange": "NSE", "quantity": -3, "product": "CNC", "view": "net"})
    margins = {"equity": {"net": 12345.5}, "commodity": {"net": 0}}
    book = kite.portfolio_from_kite(holdings, positions, margins)
    assert book.cash == 12345.5 and book.currency == "INR"
    assert [(p.ticker, p.quantity) for p in book.positions] == [("INFY.NS", 12), ("SBIN.BO", -5)]
    assert "INFY.NS" in book.render("INFY.NS")


def test_new_trades_dedupes_and_formats():
    seen: set[str] = set()
    trade = {"trade_id": "1", "exchange": "NSE", "tradingsymbol": "INFY", "transaction_type": "BUY",
             "quantity": 3, "average_price": 1500, "product": "CNC"}
    assert kite.new_trades([trade], seen) == [trade]
    assert kite.new_trades([trade], seen) == []
    msg = kite.format_trade(trade)
    assert "BUY 3 INFY" in msg and "1,500.00" in msg


def test_client_refuses_order_tools():
    client = kite.KiteMCP("http://unused")
    for tool, mode in [("orders", "place"), ("orders", "modify"), ("orders", "cancel"),
                       ("gtt", "place"), ("alerts", "create"), ("session", "logout"), ("orders", "permissions")]:
        with pytest.raises(kite.KiteError, match="read-only"):
            asyncio.run(client.call_text(tool, mode))


def test_broadcast_survives_a_failing_channel():
    got = []

    def broken(_):
        raise RuntimeError("down")

    alerts.broadcast("hi", [broken, got.append])
    assert got == ["hi"]


def test_seen_state_round_trip(tmp_path):
    path = tmp_path / "seen.json"
    kite.save_seen({"NSE:1"}, path)
    assert kite.load_seen(path) == {"NSE:1"}


def test_code_from_redirect():
    assert kite.code_from_redirect("http://localhost:8765/callback?code=abc&state=xyz") == ("abc", "xyz", None)
    with pytest.raises(kite.KiteError, match="denied"):
        kite.code_from_redirect("http://localhost:8765/callback?error=access_denied&error_description=denied")


def test_token_storage_round_trip(tmp_path):
    from mcp.shared.auth import OAuthToken

    store = kite.FileTokenStorage(tmp_path / "oauth.json")
    assert asyncio.run(store.get_tokens()) is None
    asyncio.run(store.set_tokens(OAuthToken(access_token="t", token_type="Bearer")))
    assert asyncio.run(store.get_tokens()).access_token == "t"
    assert (tmp_path / "oauth.json").stat().st_mode & 0o077 == 0
