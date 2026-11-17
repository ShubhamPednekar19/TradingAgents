"""Zerodha Kite, read through Zerodha's official Kite MCP server.

Two uses:

* ``fetch_portfolio`` turns your Kite holdings, open positions and free margin
  into a :class:`~tradingagents.portfolio.PortfolioContext`, so an analysis is
  sized against your real book.
* ``watch_trades`` polls your executed trades and sends an alert for each new
  one (console, Telegram, webhook; see ``alerts``).

Read-only by construction: the client calls only the tool modes in
``READ_ONLY_MODES`` and refuses every other one, so nothing here can place,
modify or cancel an order, whatever the server offers. You sign in on
Kite's own page (MCP OAuth), so your password never passes through this
code; the tokens are kept in ``~/.tradingagents/kite_oauth.json``. Point
``TRADINGAGENTS_KITE_MCP_URL`` at a self-hosted server to use your own
Kite Connect app instead.

Needs the MCP client: ``pip install "tradingagents[kite]"``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections.abc import Callable, Iterable
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from tradingagents.integrations.alerts import Notifier, broadcast, console_notifier
from tradingagents.portfolio import PortfolioContext, Position

logger = logging.getLogger(__name__)

DEFAULT_KITE_MCP_URL = "https://mcp.kite.trade/mcp"

# The only (tool, mode) pairs this client will ever call. The server's
# ``orders`` tool also places, modifies and cancels orders, so the guard is on
# the mode, not just the tool name; any mode not listed here is refused.
READ_ONLY_MODES = {
    "portfolio": frozenset({"profile", "margins", "holdings", "positions"}),
    "orders": frozenset({"list", "history", "trades", "all_trades"}),
    "session": frozenset({"status"}),
}

# Kite exchange -> the suffix Yahoo Finance (the default data vendor) uses.
_YAHOO_SUFFIX = {"NSE": ".NS", "BSE": ".BO"}

class KiteError(RuntimeError):
    pass


class KiteNotLoggedIn(KiteError):
    pass


# ---------------------------------------------------------------- pure helpers


def kite_mcp_url() -> str:
    return os.getenv("TRADINGAGENTS_KITE_MCP_URL", "").strip() or DEFAULT_KITE_MCP_URL


# NSE trading-series tags Kite appends (HFCL-BE); Yahoo lists the plain symbol.
_SERIES = re.compile(r"-(BE|BZ|BL|BT|SM|ST|IL|GB|GS|IT|RR|EQ)$")


def yahoo_symbol(tradingsymbol: str, exchange: str | None) -> str:
    """RELIANCE on NSE -> RELIANCE.NS. Derivatives and other venues keep Kite's spelling."""
    suffix = _YAHOO_SUFFIX.get((exchange or "").upper())
    symbol = tradingsymbol.strip().upper()
    return _SERIES.sub("", symbol) + suffix if suffix else symbol


def parse_payload(text: str) -> Any:
    """The JSON a Kite MCP tool returns, unwrapped from a ``{"data": ...}`` page if paginated."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise KiteError(f"Kite MCP returned non-JSON content: {text[:200]}") from exc
    if isinstance(value, dict) and "data" in value and set(value) <= {"data", "pagination", "status"}:
        return value["data"]
    return value


def _rows(value: Any) -> list[dict]:
    if value is None:
        return []
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    raise KiteError(f"expected a list from Kite, got {type(value).__name__}")


def _free_cash(margins: Any) -> float | None:
    """Free equity margin, from either the all-segments or the equity-only shape."""
    if not isinstance(margins, dict):
        return None
    segment = margins.get("equity", margins)
    if not isinstance(segment, dict):
        return None
    if isinstance(segment.get("net"), (int, float)):
        return float(segment["net"])
    available = segment.get("available")
    if isinstance(available, dict):
        for key in ("live_balance", "cash"):
            if isinstance(available.get(key), (int, float)):
                return float(available[key])
    return None


def portfolio_from_kite(holdings: Any, positions: Any, margins: Any = None) -> PortfolioContext:
    """Holdings (delivery, incl. T1) plus today's open net positions, and free equity margin."""
    book: list[Position] = []
    for row in _rows(holdings):
        qty = float(row.get("quantity") or 0) + float(row.get("t1_quantity") or 0)
        if qty:
            book.append(Position(
                ticker=yahoo_symbol(row["tradingsymbol"], row.get("exchange")),
                quantity=qty,
                average_price=row.get("average_price"),
            ))
    if isinstance(positions, dict):  # older shape: {"net": [...], "day": [...]}
        positions = positions.get("net", [])
    # Rows carry view=day (today's activity) or view=net (current exposure); only net is the book.
    for row in (r for r in _rows(positions) if r.get("view", "net") == "net"):
        qty = float(row.get("quantity") or 0)
        # Kite deducts today's delivery (CNC) sells from holdings already, so they
        # would count twice; today's CNC buys reach holdings only from the next day.
        if row.get("product") == "CNC" and qty < 0:
            continue
        if qty:
            book.append(Position(
                ticker=yahoo_symbol(row["tradingsymbol"], row.get("exchange")),
                quantity=qty,
                average_price=row.get("average_price"),
            ))
    return PortfolioContext(cash=_free_cash(margins), currency="INR", positions=book)


def trade_key(trade: dict) -> str:
    return f"{trade.get('exchange', '')}:{trade.get('trade_id') or trade.get('order_id')}"


def format_trade(trade: dict) -> str:
    side = str(trade.get("transaction_type", "")).upper() or "TRADE"
    qty = trade.get("quantity") or trade.get("filled_quantity")
    price = trade.get("average_price") or trade.get("price")
    when = trade.get("fill_timestamp") or trade.get("exchange_timestamp") or trade.get("order_timestamp") or ""
    price_text = f" @ ₹{float(price):,.2f}" if isinstance(price, (int, float)) else ""
    product = f" [{trade['product']}]" if trade.get("product") else ""
    return (
        f"Kite trade: {side} {qty} {trade.get('tradingsymbol', '?')} on "
        f"{trade.get('exchange', '?')}{price_text}{product} {when}".rstrip()
    )


def new_trades(trades: Iterable[dict], seen: set[str]) -> list[dict]:
    fresh = []
    for trade in trades:
        key = trade_key(trade)
        if key not in seen:
            seen.add(key)
            fresh.append(trade)
    return fresh


def _seen_path() -> Path:
    return Path.home() / ".tradingagents" / "kite_seen_trades.json"


def load_seen(path: Path | None = None) -> set[str]:
    path = path or _seen_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    # Kite trade ids reset daily, so keys from an earlier day are dropped.
    if data.get("day") != time.strftime("%Y-%m-%d"):
        return set()
    return set(data.get("keys", []))


def save_seen(seen: set[str], path: Path | None = None) -> None:
    path = path or _seen_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"day": time.strftime("%Y-%m-%d"), "keys": sorted(seen)}), encoding="utf-8")


def _print_url(url: str) -> None:
    print("Open this link to log in to Kite:\n  " + url + "\n"
          "If the page you land on does not load (the browser is on another machine), "
          "copy its full address from the address bar and paste it here.", flush=True)


# ---------------------------------------------------------------- OAuth sign-in
#
# The hosted server asks for MCP OAuth (PKCE, dynamic client registration):
# you log in on Kite's page, and the browser is redirected to a localhost URL
# carrying a one-time code. A small local server catches that redirect; where
# the browser runs on another machine, paste the redirected URL instead.

OAUTH_PORT = int(os.getenv("TRADINGAGENTS_KITE_OAUTH_PORT", "8765"))
REDIRECT_URI = f"http://localhost:{OAUTH_PORT}/callback"


def _oauth_path() -> Path:
    return Path.home() / ".tradingagents" / "kite_oauth.json"


class FileTokenStorage:
    """Keeps the OAuth client registration and tokens in a private file between runs."""

    def __init__(self, path: Path | None = None):
        self.path = path or _oauth_path()

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _write(self, key: str, value: dict) -> None:
        data = self._read()
        data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data), encoding="utf-8")
        self.path.chmod(0o600)

    async def get_tokens(self):
        from mcp.shared.auth import OAuthToken
        raw = self._read().get("tokens")
        return OAuthToken.model_validate(raw) if raw else None

    async def set_tokens(self, tokens) -> None:
        self._write("tokens", tokens.model_dump(mode="json", exclude_none=True))

    async def get_client_info(self):
        from mcp.shared.auth import OAuthClientInformationFull
        raw = self._read().get("client")
        return OAuthClientInformationFull.model_validate(raw) if raw else None

    async def set_client_info(self, client_info) -> None:
        self._write("client", client_info.model_dump(mode="json", exclude_none=True))


def code_from_redirect(url: str) -> tuple[str, str | None, str | None]:
    """The code, state and issuer from the URL the browser was redirected to."""
    from urllib.parse import parse_qs, urlparse
    query = parse_qs(urlparse(url.strip()).query)
    if "code" not in query:
        error = query.get("error_description") or query.get("error") or ["no code in that URL"]
        raise KiteError(f"Kite sign-in failed: {error[0]}")
    return query["code"][0], query.get("state", [None])[0], query.get("iss", [None])[0]


async def _wait_for_redirect(timeout: float = 600) -> str:
    """The first redirected URL to arrive, at the local callback server or pasted on stdin."""
    import sys
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    loop = asyncio.get_running_loop()
    got: asyncio.Future[str] = loop.create_future()

    def deliver(url: str) -> None:
        if not got.done():
            got.set_result(url)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - http.server API
            if not self.path.startswith("/callback"):
                self.send_response(404)
                self.end_headers()
                return
            loop.call_soon_threadsafe(deliver, f"http://localhost:{OAUTH_PORT}{self.path}")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"Kite sign-in done. You can close this tab.")

        def log_message(self, *args):
            pass

    server = None
    try:
        server = HTTPServer(("127.0.0.1", OAUTH_PORT), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
    except OSError as exc:
        logger.warning("local sign-in callback unavailable on port %s: %s", OAUTH_PORT, exc)

    def read_stdin() -> None:
        for line in sys.stdin:
            if "code=" in line or "error=" in line:
                loop.call_soon_threadsafe(deliver, line.strip())
                return

    threading.Thread(target=read_stdin, daemon=True).start()
    try:
        return await asyncio.wait_for(got, timeout)
    except TimeoutError:
        raise KiteNotLoggedIn("Kite sign-in was not completed in time") from None
    finally:
        if server:
            server.shutdown()


def _oauth(server_url: str, on_url: Callable[[str], None]):
    from mcp.client.auth import AuthorizationCodeResult, OAuthClientProvider
    from mcp.shared.auth import OAuthClientMetadata

    async def redirect(url: str) -> None:
        on_url(url)

    async def callback() -> AuthorizationCodeResult:
        code, state, iss = code_from_redirect(await _wait_for_redirect())
        return AuthorizationCodeResult(code=code, state=state, iss=iss)

    metadata = OAuthClientMetadata(
        client_name="TradingAgents (read-only)",
        redirect_uris=[REDIRECT_URI],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="none",
    )
    return OAuthClientProvider(server_url, metadata, FileTokenStorage(), redirect, callback)


# ---------------------------------------------------------------- MCP client


class KiteMCP:
    """One MCP session with the Kite server. Its Kite login lives as long as the session."""

    def __init__(self, url: str | None = None, on_url: Callable[[str], None] | None = None):
        self.url = url or kite_mcp_url()
        self.on_url = on_url or _print_url
        self._stack: AsyncExitStack | None = None
        self._session = None

    async def __aenter__(self) -> KiteMCP:
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
            from mcp.shared._httpx_utils import create_mcp_http_client
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise KiteError('the Kite integration needs the MCP client: pip install "tradingagents[kite]"') from exc
        self._stack = AsyncExitStack()
        try:
            http = await self._stack.enter_async_context(
                create_mcp_http_client(auth=_oauth(self.url, self.on_url)))
            read, write = await self._stack.enter_async_context(streamable_http_client(self.url, http_client=http))
            self._session = await self._stack.enter_async_context(ClientSession(read, write))
            await self._session.initialize()
        except BaseException as exc:
            await self._stack.aclose()
            if isinstance(exc, KiteError):
                raise
            raise KiteError(f"could not reach the Kite MCP server at {self.url}: {exc}") from exc
        return self

    async def __aexit__(self, *exc) -> None:
        if self._stack:
            await self._stack.aclose()

    async def call_text(self, tool: str, mode: str, **arguments: Any) -> str:
        if mode not in READ_ONLY_MODES.get(tool, ()):
            raise KiteError(f"refusing to call {tool} mode={mode}: this integration is read-only")
        result = await self._session.call_tool(tool, {"mode": mode, **arguments})
        text = "\n".join(getattr(part, "text", "") for part in result.content).strip()
        if getattr(result, "is_error", None) or getattr(result, "isError", False):
            if re.search(r"log ?in|session|token|auth|expired", text, re.IGNORECASE):
                raise KiteNotLoggedIn(text)
            raise KiteError(f"{tool} {mode} failed: {text}")
        return text

    async def call(self, tool: str, mode: str, **arguments: Any) -> Any:
        return parse_payload(await self.call_text(tool, mode, **arguments))

    async def login(self) -> None:
        """Fail with KiteNotLoggedIn when the Kite login behind the sign-in has expired (daily, 6 AM IST)."""
        status = await self.call("session", "status")
        if not status.get("authenticated") or status.get("kite_session_expired"):
            raise KiteNotLoggedIn("Kite session expired")

    async def portfolio(self) -> PortfolioContext:
        holdings = await self.call("portfolio", "holdings", limit=500)
        positions = await self.call("portfolio", "positions", limit=500)
        try:
            margins = await self.call("portfolio", "margins")
        except KiteError as exc:
            logger.warning("margins unavailable, cash left unset: %s", exc)
            margins = None
        return portfolio_from_kite(holdings, positions, margins)

    async def trades(self) -> list[dict]:
        return _rows(await self.call("orders", "all_trades", limit=500))


# ---------------------------------------------------------------- entry points


def forget_sign_in() -> None:
    """Drop the stored tokens so the next connection signs in to Kite afresh."""
    storage = FileTokenStorage()
    data = storage._read()
    if data.pop("tokens", None) is not None:
        storage.path.write_text(json.dumps(data), encoding="utf-8")


async def _fetch_portfolio(on_url: Callable[[str], None]) -> PortfolioContext:
    for attempt in (1, 2):
        try:
            async with KiteMCP(on_url=on_url) as kite:
                await kite.login()
                return await kite.portfolio()
        except KiteNotLoggedIn:
            if attempt == 2:
                raise
            forget_sign_in()
    raise AssertionError("unreachable")


def fetch_portfolio(on_url: Callable[[str], None] = _print_url) -> PortfolioContext:
    return asyncio.run(_fetch_portfolio(on_url))


async def _watch(notifiers: list[Notifier], interval: float, alert_existing: bool,
                 on_url: Callable[[str], None], max_polls: int | None) -> None:
    seen = load_seen()
    first = not seen and not alert_existing
    polls = 0
    while max_polls is None or polls < max_polls:
        try:
            def announce(url: str) -> None:
                on_url(url)
                broadcast(f"Kite login needed for trade alerts: {url}", [n for n in notifiers if n is not console_notifier])

            async with KiteMCP(on_url=announce) as kite:
                await kite.login()
                while max_polls is None or polls < max_polls:
                    polls += 1
                    fresh = new_trades(await kite.trades(), seen)
                    if first:
                        # Trades already done today before the watcher started are not news.
                        first = False
                    else:
                        for trade in fresh:
                            broadcast(format_trade(trade), notifiers)
                    save_seen(seen)
                    await asyncio.sleep(interval)
        except KiteNotLoggedIn as exc:
            # Kite sessions end daily (6 AM IST); sign in afresh, which sends a new link.
            logger.warning("Kite session ended: %s", exc)
            forget_sign_in()
        except Exception as exc:  # noqa: BLE001 - a network blip must not end the watch
            logger.warning("Kite watch error, retrying in %ss: %s", interval, exc)
            await asyncio.sleep(interval)


def watch_trades(notifiers: list[Notifier], interval: float = 30, alert_existing: bool = False,
                 on_url: Callable[[str], None] = _print_url, max_polls: int | None = None) -> None:
    """Poll Kite for executed trades and alert on each new one, until interrupted."""
    asyncio.run(_watch(notifiers, interval, alert_existing, on_url, max_polls))
