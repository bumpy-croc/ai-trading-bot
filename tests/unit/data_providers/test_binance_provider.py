from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import Mock, patch

import pandas as pd
import pytest

pytestmark = pytest.mark.unit

try:
    from src.data_providers.binance_provider import (
        STOP_LOSS_LIMIT_SLIPPAGE_FACTOR,
        BinanceProvider,
        with_rate_limit_retry,
    )
    from src.data_providers.exchange_interface import OrderLookupError, OrderSide

    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    BinanceProvider = Mock
    with_rate_limit_retry = None
    STOP_LOSS_LIMIT_SLIPPAGE_FACTOR = 0.005
    OrderSide = Mock
    OrderLookupError = Exception


@pytest.mark.unit
def test_binance_provider_does_not_apply_nest_asyncio():
    """Importing binance_provider must NOT monkey-patch asyncio via nest_asyncio.

    nest_asyncio.apply() made the shared TWM event loop reentrant but did not
    restore the contextvars Context, so the always-active kline socket's
    _read_ready spewed ~2,100/hr 'cannot enter context' errors. It was vestigial
    (left over from a removed custom-loop design) and was removed (#616).
    """
    import asyncio

    import src.data_providers.binance_provider  # noqa: F401 — import runs any apply()

    assert getattr(asyncio, "_nest_patched", False) is False


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
@patch("src.data_providers.binance_provider.get_binance_api_endpoint", return_value="binance")
@patch("src.data_providers.binance_provider.ThreadedWebsocketManager")
@patch("src.data_providers.binance_provider.Client")
@patch("src.data_providers.binance_provider.get_config")
def test_each_twm_gets_its_own_event_loop(mock_config, mock_client, mock_twm, mock_endpoint):
    """Two providers (kline + user streams) must not share an event loop.

    python-binance's ThreadedApiManager captures the current loop at construction.
    Two TWMs sharing the main-thread loop make the second stream's
    run_until_complete() raise 'This event loop is already running' — which stopped
    the margin user-stream from starting after nest_asyncio was removed (#646). Each
    TWM must therefore receive a distinct, freshly-created loop, with no nest_asyncio.
    """
    import asyncio

    mock_config_obj = Mock()
    mock_config_obj.get_required.return_value = "fake_key"
    mock_config.return_value = mock_config_obj

    loops = []
    try:
        for _ in range(2):
            BinanceProvider()._ensure_twm()
        assert mock_twm.call_count == 2
        loops = [call.kwargs["loop"] for call in mock_twm.call_args_list]
        assert all(isinstance(loop, asyncio.AbstractEventLoop) for loop in loops)
        assert loops[0] is not loops[1]  # distinct loops — not the shared main-thread loop
        assert getattr(asyncio, "_nest_patched", False) is False  # no nest_asyncio
    finally:
        for loop in loops:
            loop.close()


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
@patch("src.data_providers.binance_provider.Client")
@patch("src.data_providers.binance_provider.get_config")
def test_stop_streams_closes_twm_event_loop(mock_config, mock_client):
    """stop_streams() must close the per-TWM event loop so its FDs aren't leaked."""
    import asyncio

    mock_config_obj = Mock()
    mock_config_obj.get_required.return_value = "fake_key"
    mock_config.return_value = mock_config_obj

    provider = BinanceProvider()
    loop = asyncio.new_event_loop()
    provider._twm = Mock()
    provider._twm_loop = loop

    provider.stop_streams()

    assert loop.is_closed()
    assert provider._twm is None
    assert provider._twm_loop is None


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
@patch("src.data_providers.binance_provider.Client")
@patch("src.data_providers.binance_provider.get_config")
def test_kline_socket_starts_bound_to_twm_loop(mock_config, mock_client):
    """The kline socket is built synchronously on the calling thread and binds to the
    CURRENT loop; start_kline_stream must install the manager's own loop for the call
    (so the socket binds to the loop the TWM thread runs, not a loop nothing runs) and
    restore the previous loop afterwards. Regression guard for #650 — passing a per-
    manager loop without making it current bound the socket to the wrong loop and zero
    kline events were ever delivered.
    """
    import asyncio

    mock_config_obj = Mock()
    mock_config_obj.get_required.return_value = "fake_key"
    mock_config.return_value = mock_config_obj

    provider = BinanceProvider()
    twm_loop = asyncio.new_event_loop()
    provider._twm_loop = twm_loop
    provider._twm = Mock()
    seen = {}

    def fake_start_kline_socket(**kwargs):
        seen["loop_during"] = asyncio.get_event_loop()
        return "kline-key"

    provider._twm.start_kline_socket = fake_start_kline_socket

    main_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(main_loop)
    try:
        ok = provider.start_kline_stream("ETHUSDT", "1h", lambda m: None)
        assert ok is True
        # Socket constructed while the MANAGER loop was current (not main_loop), so
        # python-binance binds it to the loop the TWM thread actually runs.
        assert seen["loop_during"] is twm_loop
        # Caller's previous loop is restored (no permanent main-thread hijack).
        assert asyncio.get_event_loop() is main_loop
    finally:
        asyncio.set_event_loop(None)
        main_loop.close()
        twm_loop.close()


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
@patch("src.data_providers.binance_provider.Client")
@patch("src.data_providers.binance_provider.get_config")
def test_socket_on_twm_loop_restores_previous_loop_on_exception(mock_config, mock_client):
    """The previous loop must be restored even when start_fn raises — otherwise a
    failed socket start would leave the caller's loop hijacked (the #646 regression).
    """
    import asyncio

    mock_config_obj = Mock()
    mock_config_obj.get_required.return_value = "fake_key"
    mock_config.return_value = mock_config_obj

    provider = BinanceProvider()
    twm_loop = asyncio.new_event_loop()
    provider._twm_loop = twm_loop
    main_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(main_loop)

    def boom():
        assert asyncio.get_event_loop() is twm_loop  # manager loop installed during call
        raise RuntimeError("socket start failed")

    try:
        with pytest.raises(RuntimeError):
            provider._socket_on_twm_loop(boom)
        assert asyncio.get_event_loop() is main_loop  # restored despite the exception
    finally:
        asyncio.set_event_loop(None)
        main_loop.close()
        twm_loop.close()


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
@patch("src.data_providers.binance_provider.Client")
@patch("src.data_providers.binance_provider.get_config")
def test_user_socket_starts_bound_to_twm_loop(mock_config, mock_client):
    """The margin/user socket must bind to the manager loop too (same #650 mechanism)."""
    import asyncio

    mock_config_obj = Mock()
    mock_config_obj.get_required.return_value = "fake_key"
    mock_config.return_value = mock_config_obj

    provider = BinanceProvider()
    provider._use_margin = True
    twm_loop = asyncio.new_event_loop()
    provider._twm_loop = twm_loop
    provider._twm = Mock()
    seen = {}

    def fake_start_margin_socket(**kwargs):
        seen["loop_during"] = asyncio.get_event_loop()
        return "user-key"

    provider._twm.start_margin_socket = fake_start_margin_socket

    main_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(main_loop)
    try:
        ok = provider.start_user_stream(lambda m: None)
        assert ok is True
        assert seen["loop_during"] is twm_loop
        assert asyncio.get_event_loop() is main_loop
    finally:
        asyncio.set_event_loop(None)
        main_loop.close()
        twm_loop.close()


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
class TestBinanceDataProvider:
    @pytest.mark.data_provider
    def test_binance_provider_initialization(self):
        with patch("src.data_providers.binance_provider.get_config") as mock_config:
            mock_config_obj = Mock()
            mock_config_obj.get_required.return_value = "fake_key"
            mock_config.return_value = mock_config_obj
            provider = BinanceProvider()
            assert provider is not None

    @pytest.mark.data_provider
    @patch("src.data_providers.binance_provider.Client")
    def test_client_constructed_with_rest_timeout(self, mock_client_class):
        """Every Binance REST call gets a socket timeout via requests_params (#631).

        A timeout-less client lets a half-open TCP socket hang order polling,
        reconciliation, and the WS disconnect-recovery path indefinitely.
        """
        mock_client_class.return_value = Mock()
        with patch("src.data_providers.binance_provider.get_config") as mock_config:
            mock_config_obj = Mock()
            mock_config_obj.get_required.return_value = "fake_key"
            mock_config_obj.get_float.return_value = 15.0
            mock_config.return_value = mock_config_obj
            BinanceProvider()

        assert mock_client_class.called
        assert mock_client_class.call_args.kwargs.get("requests_params") == {"timeout": 15.0}

    @pytest.mark.data_provider
    @patch("src.data_providers.binance_provider.Client")
    def test_binance_historical_data_success(self, mock_client_class):
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_historical_klines.return_value = [
            [
                1640995200000,
                "50000",
                "50100",
                "49900",
                "50050",
                "100",
                1640995259999,
                "5000000",
                1000,
                "50",
                "2500000",
                "0",
            ],
            [
                1640998800000,
                "50050",
                "50150",
                "49950",
                "50100",
                "110",
                1640998859999,
                "5500000",
                1100,
                "55",
                "2750000",
                "0",
            ],
        ]
        with patch("src.data_providers.binance_provider.get_config") as mock_config:
            mock_config_obj = Mock()
            mock_config_obj.get_required.return_value = "fake_key"
            mock_config_obj.get_float.return_value = 60.0  # Timeout config
            mock_config.return_value = mock_config_obj
            provider = BinanceProvider()
            df = provider.get_historical_data(
                "BTCUSDT", "1h", datetime(2022, 1, 1), datetime(2022, 1, 2)
            )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert all(col in df.columns for col in ["open", "high", "low", "close", "volume"])

    @pytest.mark.data_provider
    @patch("src.data_providers.binance_provider.Client")
    def test_binance_api_error_handling(self, mock_client_class):
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_historical_klines.side_effect = Exception("API Error")
        with patch("src.data_providers.binance_provider.get_config") as mock_config:
            mock_config_obj = Mock()
            mock_config_obj.get_required.return_value = "fake_key"
            mock_config_obj.get_float.return_value = 60.0  # Timeout config
            mock_config.return_value = mock_config_obj
            provider = BinanceProvider()
            try:
                result = provider.get_historical_data("BTCUSDT", "1h", datetime(2022, 1, 1))
                assert isinstance(result, pd.DataFrame)
                assert len(result) == 0
            except Exception as e:
                assert "api" in str(e).lower() or "error" in str(e).lower()

    @pytest.mark.data_provider
    @patch("src.data_providers.binance_provider.Client")
    def test_binance_rate_limit_handling(self, mock_client_class):
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        try:
            from binance.exceptions import BinanceAPIException

            mock_response = Mock()
            mock_response.text = '{"code": -1003, "msg": "Rate limit exceeded"}'
            exception_to_raise = BinanceAPIException(
                mock_response, status_code=429, text=mock_response.text
            )
        except (ImportError, TypeError, AttributeError):
            exception_to_raise = Exception("Rate limit exceeded")
        mock_client.get_historical_klines.side_effect = exception_to_raise
        with patch("src.data_providers.binance_provider.get_config") as mock_config:
            mock_config_obj = Mock()
            mock_config_obj.get_required.return_value = "fake_key"
            mock_config_obj.get_float.return_value = 60.0  # Timeout config
            mock_config.return_value = mock_config_obj
            provider = BinanceProvider()
            try:
                result = provider.get_historical_data("BTCUSDT", "1h", datetime(2022, 1, 1))
                assert isinstance(result, pd.DataFrame)
                assert len(result) == 0
            except Exception as e:
                assert any(s in str(e).lower() for s in ["rate limit", "exceeded", "error"])

    @pytest.mark.data_provider
    @patch("src.data_providers.binance_provider.Client")
    def test_binance_data_validation(self, mock_client_class):
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        with patch("src.data_providers.binance_provider.get_config") as mock_config:
            mock_config_obj = Mock()
            mock_config_obj.get_required.return_value = "fake_key"
            mock_config_obj.get_float.return_value = 60.0  # Timeout config
            mock_config.return_value = mock_config_obj
            provider = BinanceProvider()
            try:
                result = provider.get_historical_data("BTCUSDT", "invalid", datetime.now(UTC))
                assert isinstance(result, pd.DataFrame)
                assert len(result) == 0
            except (ValueError, Exception) as e:
                assert any(s in str(e).lower() for s in ["invalid", "timeframe", "error"])


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
class TestRateLimitRetryDecorator:
    """Tests for the with_rate_limit_retry decorator."""

    def test_successful_call_no_retry(self):
        """Verify decorated function returns immediately on success."""
        # Arrange
        call_count = 0

        @with_rate_limit_retry(max_retries=3, base_delay=0.01)
        def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        # Act
        result = success_func()

        # Assert
        assert result == "success"
        assert call_count == 1

    @patch("src.data_providers.binance_provider.time.sleep")
    def test_retry_on_rate_limit_error(self, mock_sleep):
        """Verify retry with exponential backoff on rate limit errors."""
        # Arrange
        from binance.exceptions import BinanceAPIException

        call_count = 0

        @with_rate_limit_retry(max_retries=3, base_delay=1.0)
        def rate_limited_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                exc = BinanceAPIException(Mock(), 429, "Rate limited")
                exc.code = -1003  # Rate limit error code
                raise exc
            return "success"

        # Act
        result = rate_limited_then_success()

        # Assert
        assert result == "success"
        assert call_count == 3
        assert mock_sleep.call_count == 2
        # Verify exponential backoff: 1s, 2s
        mock_sleep.assert_any_call(1.0)
        mock_sleep.assert_any_call(2.0)

    @patch("src.data_providers.binance_provider.time.sleep")
    def test_max_retries_exhausted(self, mock_sleep):
        """Verify exception raised after max retries exhausted."""
        # Arrange
        from binance.exceptions import BinanceAPIException

        @with_rate_limit_retry(max_retries=2, base_delay=0.01)
        def always_rate_limited():
            exc = BinanceAPIException(Mock(), 429, "Rate limited")
            exc.code = -1003
            raise exc

        # Act & Assert
        with pytest.raises(BinanceAPIException):
            always_rate_limited()
        assert mock_sleep.call_count == 2

    def test_non_rate_limit_error_not_retried(self):
        """Verify non-rate-limit errors are raised immediately."""
        # Arrange
        from binance.exceptions import BinanceAPIException

        call_count = 0

        @with_rate_limit_retry(max_retries=3, base_delay=0.01)
        def other_error():
            nonlocal call_count
            call_count += 1
            exc = BinanceAPIException(Mock(), 400, "Bad request")
            exc.code = -1000  # Not a rate limit code
            raise exc

        # Act & Assert
        with pytest.raises(BinanceAPIException):
            other_error()
        assert call_count == 1

    def test_non_binance_exception_not_retried(self):
        """Verify non-Binance exceptions are raised immediately."""
        # Arrange
        call_count = 0

        @with_rate_limit_retry(max_retries=3, base_delay=0.01)
        def generic_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Something went wrong")

        # Act & Assert
        with pytest.raises(ValueError):
            generic_error()
        assert call_count == 1


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
@patch("src.data_providers.binance_provider.BINANCE_AVAILABLE", True)
class TestPlaceStopLossOrder:
    """Tests for the place_stop_loss_order method."""

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_place_stop_loss_sell_order_success(self, mock_config, mock_client_class):
        """Verify successful stop-loss sell order placement."""
        # Arrange
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj

        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.create_order.return_value = {"orderId": "12345"}
        mock_client.get_exchange_info.return_value = {"symbols": []}

        provider = BinanceProvider()
        provider.get_symbol_info = Mock(
            return_value={"step_size": 0.0001, "tick_size": 0.01, "base_asset": "BTC"}
        )
        provider.get_balance = Mock(return_value=Mock(free=1.0))

        # Act
        result = provider.place_stop_loss_order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            quantity=0.1,
            stop_price=50000.0,
        )

        # Assert
        assert result == "12345"
        mock_client.create_order.assert_called_once()
        call_args = mock_client.create_order.call_args
        assert call_args.kwargs["type"] == "STOP_LOSS_LIMIT"
        assert call_args.kwargs["side"] == "SELL"
        assert call_args.kwargs["timeInForce"] == "GTC"

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_place_stop_loss_buy_order_success(self, mock_config, mock_client_class):
        """Verify successful stop-loss buy order placement."""
        # Arrange
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj

        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.create_order.return_value = {"orderId": "67890"}
        mock_client.get_exchange_info.return_value = {"symbols": []}

        provider = BinanceProvider()
        provider.get_symbol_info = Mock(
            return_value={"step_size": 0.0001, "tick_size": 0.01, "base_asset": "BTC"}
        )

        # Act
        result = provider.place_stop_loss_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=0.1,
            stop_price=50000.0,
        )

        # Assert
        assert result == "67890"
        call_args = mock_client.create_order.call_args
        assert call_args.kwargs["side"] == "BUY"

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_sell_stop_loss_capped_at_free_balance(self, mock_config, mock_client_class):
        """A SELL SL is capped at the free base balance + rounded DOWN, avoiding -2010.

        Binance deducts the trade fee from a buy's fill, so the tracked position
        quantity can exceed the free ETH; without the cap the SL was rejected with
        -2010 and the position was left unprotected.
        """
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.create_order.return_value = {"orderId": "sl1"}
        provider = BinanceProvider()
        provider.get_symbol_info = Mock(
            return_value={"step_size": 0.0001, "tick_size": 0.01, "base_asset": "ETH"}
        )
        # Free ETH is slightly under the tracked position (fee taken from the buy fill).
        provider.get_balance = Mock(return_value=Mock(free=0.004995))

        result = provider.place_stop_loss_order(
            symbol="ETHUSDT", side=OrderSide.SELL, quantity=0.005, stop_price=1900.0
        )

        assert result == "sl1"
        provider.get_balance.assert_called_once_with("ETH")
        # min(0.005, 0.004995)=0.004995 -> floor(/0.0001)*0.0001 = 0.0049 (never > holdings)
        # Sent as a fixed-point string (not a raw float) so urlencode() never renders
        # it in scientific notation (#745).
        sent_qty = mock_client.create_order.call_args.kwargs["quantity"]
        assert isinstance(sent_qty, str)
        assert float(sent_qty) == pytest.approx(0.0049, abs=1e-9)

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_sell_stop_loss_zero_free_balance_skips_order(self, mock_config, mock_client_class):
        """No free base asset to protect -> return None, place nothing."""
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        provider = BinanceProvider()
        provider.get_symbol_info = Mock(return_value={"step_size": 0.0001, "tick_size": 0.01})
        provider.get_balance = Mock(return_value=Mock(free=0.0))

        result = provider.place_stop_loss_order(
            symbol="ETHUSDT", side=OrderSide.SELL, quantity=0.005, stop_price=1900.0
        )

        assert result is None
        mock_client.create_order.assert_not_called()

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_buy_stop_loss_not_capped_by_base_balance(self, mock_config, mock_client_class):
        """A BUY (short cover) SL is funded from quote, so it is NOT capped by base holdings."""
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.create_order.return_value = {"orderId": "sl2"}
        provider = BinanceProvider()
        provider.get_symbol_info = Mock(return_value={"step_size": 0.0001, "tick_size": 0.01})
        provider.get_balance = Mock()

        result = provider.place_stop_loss_order(
            symbol="ETHUSDT", side=OrderSide.BUY, quantity=0.005, stop_price=2100.0
        )

        assert result == "sl2"
        provider.get_balance.assert_not_called()

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_sell_stop_loss_fails_closed_without_symbol_info(self, mock_config, mock_client_class):
        """A transient get_symbol_info failure must refuse to place, not send raw floats.

        Without symbol_info there is no known tick/lot precision, so quantizing
        is impossible — sending the order anyway means an unquantized
        price/quantity that Binance rejects with -1111/51077, leaving the
        position unprotected with no loud error (#1126). The order must not be
        sent, and the refusal must be recorded via the durable order-error sink.

        Supersedes the pre-#1126 `test_sell_stop_loss_caps_without_symbol_info`,
        which asserted the opposite for a small (0.2%) shortfall: that a capped,
        unrounded order was still sent. That is exactly the silently-wrong-order
        shape #1126 exists to remove — no shortfall size is small enough to skip
        quantization, since quantization itself is what's impossible without
        symbol_info. There is no longer a "small enough to proceed" case.
        """
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        provider = BinanceProvider()
        provider.get_symbol_info = Mock(return_value=None)  # transient API failure
        provider.get_balance = Mock(return_value=Mock(free=0.004))
        recorded = []
        provider.order_error_sink = recorded.append

        result = provider.place_stop_loss_order(
            symbol="ETHUSDT", side=OrderSide.SELL, quantity=0.005, stop_price=1900.0
        )

        assert result is None
        mock_client.create_order.assert_not_called()
        assert len(recorded) == 1
        assert recorded[0].error_type == "SymbolInfoUnavailable"
        assert recorded[0].params["symbol_info_available"] is False

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_sell_stop_loss_fails_closed_records_the_actually_requested_price(
        self, mock_config, mock_client_class
    ):
        """The durable SymbolInfoUnavailable row must record the limit price that
        was actually being requested, not the pre-computation placeholder from
        before the auto-computed limit_price (stop_price adjusted by the slippage
        factor) was filled in."""
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        provider = BinanceProvider()
        provider.get_symbol_info = Mock(return_value=None)
        provider.get_balance = Mock(return_value=Mock(free=0.004))
        recorded = []
        provider.order_error_sink = recorded.append

        provider.place_stop_loss_order(
            symbol="ETHUSDT", side=OrderSide.SELL, quantity=0.005, stop_price=1900.0
        )

        assert len(recorded) == 1
        assert recorded[0].params["price"] is not None
        assert recorded[0].params["price"] < 1900.0  # SELL limit sits below the stop

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_sell_stop_loss_places_uncapped_when_balance_unreadable(
        self, mock_config, mock_client_class
    ):
        """If the free balance can't be read, don't block the SL — place it uncapped.

        A failed balance read must not leave the position unprotected; fall back to
        the (lot-rounded) requested quantity rather than skipping the order.
        """
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.create_order.return_value = {"orderId": "sl4"}
        provider = BinanceProvider()
        provider.get_symbol_info = Mock(
            return_value={"step_size": 0.0001, "tick_size": 0.01, "base_asset": "ETH"}
        )
        provider.get_balance = Mock(side_effect=RuntimeError("balance read failed"))

        result = provider.place_stop_loss_order(
            symbol="ETHUSDT", side=OrderSide.SELL, quantity=0.005, stop_price=1900.0
        )

        assert result == "sl4"
        sent_qty = mock_client.create_order.call_args.kwargs["quantity"]
        assert isinstance(sent_qty, str)
        assert float(sent_qty) == pytest.approx(0.005, abs=1e-9)

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_sell_stop_loss_not_capped_when_balance_sufficient(
        self, mock_config, mock_client_class
    ):
        """When free balance comfortably covers the position, no cap is applied."""
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.create_order.return_value = {"orderId": "sl5"}
        provider = BinanceProvider()
        provider.get_symbol_info = Mock(
            return_value={"step_size": 0.0001, "tick_size": 0.01, "base_asset": "ETH"}
        )
        provider.get_balance = Mock(return_value=Mock(free=1.0))

        result = provider.place_stop_loss_order(
            symbol="ETHUSDT", side=OrderSide.SELL, quantity=0.005, stop_price=1900.0
        )

        assert result == "sl5"
        sent_qty = mock_client.create_order.call_args.kwargs["quantity"]
        assert isinstance(sent_qty, str)
        assert float(sent_qty) == pytest.approx(0.005, abs=1e-9)

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_sell_stop_loss_floor_preserves_exact_lot(self, mock_config, mock_client_class):
        """floor(qty/step) must not shed a whole lot to float noise (0.29, not 0.28)."""
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.create_order.return_value = {"orderId": "sl6"}
        provider = BinanceProvider()
        provider.get_symbol_info = Mock(
            return_value={"step_size": 0.01, "tick_size": 0.01, "base_asset": "ETH"}
        )
        provider.get_balance = Mock(return_value=Mock(free=1.0))

        result = provider.place_stop_loss_order(
            symbol="ETHUSDT", side=OrderSide.SELL, quantity=0.29, stop_price=1900.0
        )

        assert result == "sl6"
        # 0.29 / 0.01 = 28.9999996 in float; a naive floor -> 0.28. Epsilon keeps 0.29.
        sent_qty = mock_client.create_order.call_args.kwargs["quantity"]
        assert isinstance(sent_qty, str)
        assert float(sent_qty) == pytest.approx(0.29, abs=1e-9)

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_stop_loss_accepts_decimal_inputs(self, mock_config, mock_client_class):
        """Decimal stop_price/quantity (DB Numeric columns) must not raise Decimal*float.

        A DB-loaded position carries Decimal fields; without coercion the limit-price
        and lot arithmetic raised 'unsupported operand type(s) for *' and left the
        position unprotected (periodic reconciler).
        """
        from decimal import Decimal

        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.create_order.return_value = {"orderId": "sl7"}
        provider = BinanceProvider()
        provider.get_symbol_info = Mock(
            return_value={"step_size": 0.0001, "tick_size": 0.01, "base_asset": "ETH"}
        )
        provider.get_balance = Mock(return_value=Mock(free=1.0))

        result = provider.place_stop_loss_order(
            symbol="ETHUSDT",
            side=OrderSide.SELL,
            quantity=Decimal("0.005"),
            stop_price=Decimal("1900.0"),
        )

        assert result == "sl7"
        sent_qty = mock_client.create_order.call_args.kwargs["quantity"]
        assert isinstance(sent_qty, str)
        assert float(sent_qty) == pytest.approx(0.005, abs=1e-9)

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_auto_limit_price_calculation_sell(self, mock_config, mock_client_class):
        """Verify limit price is calculated below stop for sell orders."""
        # Arrange
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj

        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.create_order.return_value = {"orderId": "123"}
        mock_client.get_exchange_info.return_value = {"symbols": []}

        provider = BinanceProvider()
        provider.get_symbol_info = Mock(
            return_value={"step_size": 0.0001, "tick_size": 0.01, "base_asset": "BTC"}
        )
        provider.get_balance = Mock(return_value=Mock(free=1.0))
        stop_price = 50000.0

        # Act
        provider.place_stop_loss_order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            quantity=0.1,
            stop_price=stop_price,
        )

        # Assert
        call_args = mock_client.create_order.call_args
        limit_price = float(call_args.kwargs["price"])
        expected_limit = stop_price * (1 - STOP_LOSS_LIMIT_SLIPPAGE_FACTOR)
        assert abs(limit_price - expected_limit) < 0.01

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_auto_limit_price_calculation_buy(self, mock_config, mock_client_class):
        """Verify limit price is calculated above stop for buy orders."""
        # Arrange
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj

        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.create_order.return_value = {"orderId": "123"}
        mock_client.get_exchange_info.return_value = {"symbols": []}

        provider = BinanceProvider()
        provider.get_symbol_info = Mock(
            return_value={"step_size": 0.0001, "tick_size": 0.01, "base_asset": "BTC"}
        )
        stop_price = 50000.0

        # Act
        provider.place_stop_loss_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=0.1,
            stop_price=stop_price,
        )

        # Assert
        call_args = mock_client.create_order.call_args
        limit_price = float(call_args.kwargs["price"])
        expected_limit = stop_price * (1 + STOP_LOSS_LIMIT_SLIPPAGE_FACTOR)
        assert abs(limit_price - expected_limit) < 0.01

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_custom_limit_price_used(self, mock_config, mock_client_class):
        """Verify custom limit price overrides auto-calculation."""
        # Arrange
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj

        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.create_order.return_value = {"orderId": "123"}
        mock_client.get_exchange_info.return_value = {"symbols": []}

        provider = BinanceProvider()
        provider.get_symbol_info = Mock(
            return_value={"step_size": 0.0001, "tick_size": 0.01, "base_asset": "BTC"}
        )
        provider.get_balance = Mock(return_value=Mock(free=1.0))

        # Act
        provider.place_stop_loss_order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            quantity=0.1,
            stop_price=50000.0,
            limit_price=49500.0,
        )

        # Assert
        call_args = mock_client.create_order.call_args
        limit_price = float(call_args.kwargs["price"])
        assert abs(limit_price - 49500.0) < 0.01

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_order_exception_returns_none(self, mock_config, mock_client_class):
        """Verify BinanceOrderException returns None."""
        # Arrange
        from binance.exceptions import BinanceOrderException

        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj

        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.create_order.side_effect = BinanceOrderException(Mock(), "Order rejected")
        mock_client.get_exchange_info.return_value = {"symbols": []}

        provider = BinanceProvider()
        provider.get_symbol_info = Mock(
            return_value={"step_size": 0.0001, "tick_size": 0.01, "base_asset": "BTC"}
        )
        provider.get_balance = Mock(return_value=Mock(free=1.0))

        # Act
        result = provider.place_stop_loss_order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            quantity=0.1,
            stop_price=50000.0,
        )

        # Assert
        assert result is None

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_generic_exception_returns_none(self, mock_config, mock_client_class):
        """Verify generic exception returns None."""
        # Arrange
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj

        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.create_order.side_effect = Exception("Network error")
        mock_client.get_exchange_info.return_value = {"symbols": []}

        provider = BinanceProvider()
        provider.get_symbol_info = Mock(
            return_value={"step_size": 0.0001, "tick_size": 0.01, "base_asset": "BTC"}
        )
        provider.get_balance = Mock(return_value=Mock(free=1.0))

        # Act
        result = provider.place_stop_loss_order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            quantity=0.1,
            stop_price=50000.0,
        )

        # Assert
        assert result is None

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_missing_order_id_returns_none(self, mock_config, mock_client_class):
        """Verify missing orderId in response returns None."""
        # Arrange
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj

        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.create_order.return_value = {}  # No orderId
        mock_client.get_exchange_info.return_value = {"symbols": []}

        provider = BinanceProvider()
        provider.get_symbol_info = Mock(
            return_value={"step_size": 0.0001, "tick_size": 0.01, "base_asset": "BTC"}
        )
        provider.get_balance = Mock(return_value=Mock(free=1.0))

        # Act
        result = provider.place_stop_loss_order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            quantity=0.1,
            stop_price=50000.0,
        )

        # Assert
        assert result is None

    @patch("src.data_providers.binance_provider.time.sleep")
    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_rate_limit_1015_is_retried_by_decorator_and_succeeds(
        self, mock_config, mock_client_class, mock_sleep
    ):
        """#738: -1015 must reach with_rate_limit_retry and actually be retried.

        Before the fix, place_stop_loss_order caught BinanceAPIException
        generically and returned None on the first attempt, so the
        @with_rate_limit_retry(ban_safe=True) decorator wrapping this method
        never saw the exception and never retried — create_order was called
        exactly once and the method returned None instead of the order id.
        """
        from binance.exceptions import BinanceAPIException

        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj

        mock_client = Mock()
        mock_client_class.return_value = mock_client
        rate_limit_error = BinanceAPIException(Mock(status_code=418, headers={}), 418, "")
        rate_limit_error.code = -1015
        mock_client.create_order.side_effect = [
            rate_limit_error,
            {"orderId": "12345"},
        ]
        mock_client.get_exchange_info.return_value = {"symbols": []}

        provider = BinanceProvider()
        provider.get_symbol_info = Mock(
            return_value={"step_size": 0.0001, "tick_size": 0.01, "base_asset": "BTC"}
        )
        provider.get_balance = Mock(return_value=Mock(free=1.0))

        result = provider.place_stop_loss_order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            quantity=0.1,
            stop_price=50000.0,
        )

        assert result == "12345"
        assert mock_client.create_order.call_count == 2
        mock_sleep.assert_called_once()

    @patch("src.data_providers.binance_provider.time.sleep")
    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_rate_limit_1003_raises_immediately_without_retry(
        self, mock_config, mock_client_class, mock_sleep
    ):
        """#738: -1003 (IP ban) must propagate distinctly, not return None.

        ban_safe=True on this method's decorator means -1003 is raised
        immediately rather than retried (sleeping through an IP ban would
        leave a position's stop-loss placement blocked for minutes). Before
        the fix, place_stop_loss_order swallowed this into a plain None
        return, indistinguishable from any other failure and never reaching
        the decorator at all.
        """
        from binance.exceptions import BinanceAPIException

        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj

        mock_client = Mock()
        mock_client_class.return_value = mock_client
        ban_error = BinanceAPIException(Mock(status_code=418, headers={}), 418, "")
        ban_error.code = -1003
        mock_client.create_order.side_effect = ban_error
        mock_client.get_exchange_info.return_value = {"symbols": []}

        provider = BinanceProvider()
        provider.get_symbol_info = Mock(
            return_value={"step_size": 0.0001, "tick_size": 0.01, "base_asset": "BTC"}
        )
        provider.get_balance = Mock(return_value=Mock(free=1.0))

        with pytest.raises(BinanceAPIException):
            provider.place_stop_loss_order(
                symbol="BTCUSDT",
                side=OrderSide.SELL,
                quantity=0.1,
                stop_price=50000.0,
            )

        assert mock_client.create_order.call_count == 1
        mock_sleep.assert_not_called()


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
@patch("src.data_providers.binance_provider.BINANCE_AVAILABLE", True)
class TestPlaceOrderRateLimitDefinitiveReject:
    """#738: a rate-limit rejection on entry placement must be definitive.

    Before the fix, -1003/-1015 fell into place_order's generic "ambiguous"
    BinanceAPIException branch and returned None — the same signal used for
    a genuinely ambiguous network/timeout error. Callers (execution_engine.py)
    treat a None return as "order may have been placed", tracking a phantom
    position and entering close-only mode until a manual restart. Since the
    exchange definitively rejected the order (it was never placed), this
    must instead raise ValueError like every other definitive reject.
    """

    @pytest.mark.parametrize("error_code", [-1003, -1015])
    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_rate_limit_code_raises_value_error_not_none(
        self, mock_config, mock_client_class, error_code
    ):
        from binance.exceptions import BinanceAPIException

        from src.data_providers.exchange_interface import OrderType

        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj

        mock_client = Mock()
        mock_client_class.return_value = mock_client
        error = BinanceAPIException(Mock(status_code=418, headers={}), 418, "")
        error.code = error_code
        error.message = "Too many requests"
        mock_client.create_order.side_effect = error
        mock_client.get_exchange_info.return_value = {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "baseAsset": "BTC",
                    "quoteAsset": "USDT",
                    "status": "TRADING",
                    "filters": [
                        {"filterType": "LOT_SIZE", "minQty": "0.00001", "stepSize": "0.00001"},
                        {"filterType": "PRICE_FILTER", "minPrice": "0.01", "tickSize": "0.01"},
                        {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                    ],
                }
            ]
        }

        provider = BinanceProvider()

        with pytest.raises(ValueError, match=f"Order rejected by exchange.*{error_code}"):
            provider.place_order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0.001,
            )


def _exchange_info_for(symbol: str) -> dict:
    return {
        "symbols": [
            {
                "symbol": symbol,
                "baseAsset": symbol[:-4],
                "quoteAsset": "USDT",
                "status": "TRADING",
                "filters": [
                    {"filterType": "LOT_SIZE", "minQty": "0.00001", "stepSize": "0.00001"},
                    {"filterType": "PRICE_FILTER", "minPrice": "0.01", "tickSize": "0.01"},
                    {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                ],
            }
        ]
    }


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
@patch("src.data_providers.binance_provider.BINANCE_AVAILABLE", True)
class TestGetSymbolInfoCache:
    """#1155: a transient get_exchange_info failure must not make a previously
    known-good symbol look unknown -- only a symbol that has NEVER been fetched
    successfully should return falsy on failure.
    """

    @pytest.mark.fast
    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_transient_failure_on_warm_cache_returns_last_known_good(
        self, mock_config, mock_client_class
    ):
        """A symbol fetched successfully once must survive a later transient failure.

        This is the discriminating case for #1155: pre-fix, get_symbol_info
        returns None/falsy on ANY exception regardless of prior successful
        fetches, which flows straight into place_stop_loss_order's #1126
        fail-closed guard and forces an emergency-close on a routine blip.
        """
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_client.get_exchange_info.return_value = _exchange_info_for("BTCUSDT")
        provider = BinanceProvider()

        # Warm the cache with a successful fetch.
        first = provider.get_symbol_info("BTCUSDT")
        assert first is not None
        assert first["step_size"] == pytest.approx(0.00001)

        # Simulate a transient rate-limit / upstream 5xx blip.
        mock_client.get_exchange_info.side_effect = ConnectionError("rate limited")

        result = provider.get_symbol_info("BTCUSDT")

        assert result is not None, (
            "a warm-cache symbol must fall back to the last known-good filters "
            "on a transient failure, not fail closed"
        )
        assert result["step_size"] == pytest.approx(0.00001)
        assert result["tick_size"] == pytest.approx(0.01)

    @pytest.mark.fast
    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_never_fetched_symbol_fails_closed_on_first_failure(
        self, mock_config, mock_client_class
    ):
        """A symbol with no prior successful fetch still fails closed (#1126 intact)."""
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_exchange_info.side_effect = ConnectionError("rate limited")

        provider = BinanceProvider()

        result = provider.get_symbol_info("BTCUSDT")

        assert result is None

    @pytest.mark.fast
    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_cache_is_per_symbol(self, mock_config, mock_client_class):
        """A warm cache entry for one symbol must not mask a never-fetched other symbol."""
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_client.get_exchange_info.return_value = _exchange_info_for("BTCUSDT")
        provider = BinanceProvider()
        assert provider.get_symbol_info("BTCUSDT") is not None

        mock_client.get_exchange_info.side_effect = ConnectionError("rate limited")

        assert provider.get_symbol_info("ETHUSDT") is None

    @pytest.mark.fast
    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_successful_refetch_updates_cache(self, mock_config, mock_client_class):
        """A live fetch is always attempted -- the cache reflects the latest filters,
        not just the first ones ever seen (this is a fallback cache, not a static one).
        """
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_client.get_exchange_info.return_value = _exchange_info_for("BTCUSDT")
        provider = BinanceProvider()
        assert provider.get_symbol_info("BTCUSDT")["step_size"] == pytest.approx(0.00001)

        updated = _exchange_info_for("BTCUSDT")
        updated["symbols"][0]["filters"][0]["stepSize"] = "0.001"
        mock_client.get_exchange_info.return_value = updated

        result = provider.get_symbol_info("BTCUSDT")
        assert result["step_size"] == pytest.approx(0.001)

    @pytest.mark.fast
    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_confirmed_delisting_is_not_resurrected_by_a_later_blip(
        self, mock_config, mock_client_class
    ):
        """A successful lookup that confirms a symbol is genuinely gone (delisted)
        must drop any stale cache entry -- otherwise a later transient failure
        would serve the stale filters back out via the fallback-on-failure path,
        contradicting what Binance just told us (code-reviewer finding on #1155).
        """
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Warm the cache.
        mock_client.get_exchange_info.return_value = _exchange_info_for("BTCUSDT")
        provider = BinanceProvider()
        assert provider.get_symbol_info("BTCUSDT") is not None

        # A successful lookup that no longer lists the symbol at all (delisted).
        mock_client.get_exchange_info.return_value = {"symbols": []}
        assert provider.get_symbol_info("BTCUSDT") is None

        # A subsequent transient failure must NOT resurrect the stale entry.
        mock_client.get_exchange_info.side_effect = ConnectionError("rate limited")
        result = provider.get_symbol_info("BTCUSDT")

        assert result is None, (
            "a confirmed-absent symbol must stay absent on a later transient "
            "failure, not serve back the pre-delisting cached filters"
        )

    @pytest.mark.fast
    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_cache_key_is_normalized_across_symbol_casing(self, mock_config, mock_client_class):
        """The cache must key on the normalized exchange symbol, not the caller's raw
        string -- otherwise get_symbol_info("btcusdt") warming the cache would miss a
        later get_symbol_info("BTCUSDT") lookup during a blip and fail closed anyway.
        """
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_client.get_exchange_info.return_value = _exchange_info_for("BTCUSDT")
        provider = BinanceProvider()
        assert provider.get_symbol_info("btcusdt") is not None

        mock_client.get_exchange_info.side_effect = ConnectionError("rate limited")
        result = provider.get_symbol_info("BTCUSDT")

        assert result is not None, (
            "a differently-cased lookup for the same symbol must hit the same "
            "warm cache entry on a transient failure"
        )


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
@patch("src.data_providers.binance_provider.BINANCE_AVAILABLE", True)
class TestStopLossQuantityStepPrecision:
    """The quantity sent to Binance must carry no more decimals than LOT_SIZE allows.

    `(integer) * step_size` in float math leaves artifacts (e.g.
    round(0.0003 / 0.0001) * 0.0001 == 0.00030000000000000003). Sent verbatim, that
    over-precise value is rejected with code 51077 ("Precision is over the maximum
    defined for this asset"), which in prod made the SL submission ambiguous and left
    a position unprotected in close-only mode. The quantize step must clamp the value
    to the step's decimal count. Without the fix these assertions FAIL (the raw float
    has exponent -17 to -20); with it the exponent never goes below what the step allows.
    """

    @staticmethod
    def _make_provider(mock_config, mock_client_class, step_size, base_asset="ETH"):
        """Build a provider whose symbol_info reports `step_size` and has free balance."""
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.create_order.return_value = {"orderId": "slp"}
        provider = BinanceProvider()
        provider.get_symbol_info = Mock(
            return_value={
                "step_size": step_size,
                "tick_size": 0.01,
                "base_asset": base_asset,
            }
        )
        # Free balance comfortably covers the quantity so the SELL cap never trims it.
        provider.get_balance = Mock(return_value=Mock(free=1_000_000.0))
        return provider, mock_client

    @pytest.mark.fast
    @pytest.mark.parametrize(
        ("step_size", "quantity"),
        [
            # step -> a clean lot-multiple qty whose float round-trip is over-precise.
            (0.0001, 0.0003),  # reproduces the 51077 incident case; raw exp == -20
            (0.001, 0.009),  # raw exp == -18
            (0.01, 0.35),  # raw exp == -17
            (1.0, 40.0),  # integer step: must stay whole, no artifact introduced
        ],
    )
    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_buy_branch_quantity_quantized_to_step(
        self, mock_config, mock_client_class, step_size, quantity
    ):
        """BUY/other branch: round(qty/step)*step is quantized to the step's precision."""
        provider, mock_client = self._make_provider(mock_config, mock_client_class, step_size)

        result = provider.place_stop_loss_order(
            symbol="ETHUSDT", side=OrderSide.BUY, quantity=quantity, stop_price=2100.0
        )

        assert result == "slp"
        sent_qty = mock_client.create_order.call_args.kwargs["quantity"]
        # Sent as a fixed-point string, never scientific notation (#745).
        assert isinstance(sent_qty, str)
        step_decimals = max(0, -Decimal(str(step_size)).as_tuple().exponent)
        # No more decimal places than the step implies (exponent not below -step_decimals).
        assert Decimal(sent_qty).as_tuple().exponent >= -step_decimals
        # The quantize preserves the intended quantity (rounding artifact only).
        assert float(sent_qty) == pytest.approx(quantity, abs=step_size / 2)

    @pytest.mark.fast
    @pytest.mark.parametrize(
        ("step_size", "quantity"),
        [
            (0.0001, 0.0003),
            (0.001, 0.009),
            (0.01, 0.35),
            (1.0, 40.0),
        ],
    )
    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_sell_branch_quantity_quantized_to_step(
        self, mock_config, mock_client_class, step_size, quantity
    ):
        """SELL branch: floor(qty/step)*step is quantized to the step's precision too."""
        provider, mock_client = self._make_provider(mock_config, mock_client_class, step_size)

        result = provider.place_stop_loss_order(
            symbol="ETHUSDT", side=OrderSide.SELL, quantity=quantity, stop_price=1900.0
        )

        assert result == "slp"
        sent_qty = mock_client.create_order.call_args.kwargs["quantity"]
        assert isinstance(sent_qty, str)
        step_decimals = max(0, -Decimal(str(step_size)).as_tuple().exponent)
        assert Decimal(sent_qty).as_tuple().exponent >= -step_decimals
        assert float(sent_qty) == pytest.approx(quantity, abs=step_size / 2)

    @pytest.mark.fast
    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_reproduces_51077_case_exponent_within_four_decimals(
        self, mock_config, mock_client_class
    ):
        """Explicit 51077 reproduction: step 0.0001, qty 0.0003 -> exponent >= -4."""
        provider, mock_client = self._make_provider(mock_config, mock_client_class, 0.0001)

        result = provider.place_stop_loss_order(
            symbol="ETHUSDT", side=OrderSide.BUY, quantity=0.0003, stop_price=2100.0
        )

        assert result == "slp"
        sent_qty = mock_client.create_order.call_args.kwargs["quantity"]
        assert isinstance(sent_qty, str)
        assert Decimal(sent_qty).as_tuple().exponent >= -4
        assert float(sent_qty) == pytest.approx(0.0003, abs=1e-9)


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
@patch("src.data_providers.binance_provider.BINANCE_AVAILABLE", True)
class TestQuantityScientificNotationAvoided:
    """A quantity in [1e-5, 1e-4) must never reach the exchange as scientific notation.

    python-binance urlencodes order params. `urlencode({"quantity": 0.00009})` renders
    the value with Python's default float-to-str conversion, which switches to
    scientific notation below 1e-4 (`str(0.00009) == "9e-05"`). Binance then rejects
    the request with -1100 ("illegal characters"), which independently breaks entries,
    stop-loss placement, and closes (#745). Without the fix, the "9e-05"/"e-05"
    assertions below FAIL because the raw float is sent unformatted; with it, quantity
    is always a plain fixed-point string.
    """

    @pytest.mark.fast
    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_place_order_small_quantity_not_scientific_notation(
        self, mock_config, mock_client_class
    ):
        """place_order (entries/closes) formats a sub-1e-4 quantity as fixed-point."""
        from src.data_providers.exchange_interface import OrderType

        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.create_order.return_value = {
            "orderId": "1",
            "status": "FILLED",
            "origQty": "0.00009",
            "executedQty": "0.00009",
            "cummulativeQuoteQty": "5.0",
            "fills": [],
        }
        mock_client.get_exchange_info.return_value = {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "baseAsset": "BTC",
                    "quoteAsset": "USDT",
                    "status": "TRADING",
                    "filters": [
                        {"filterType": "LOT_SIZE", "minQty": "0.00001", "stepSize": "0.00001"},
                        {"filterType": "PRICE_FILTER", "minPrice": "0.01", "tickSize": "0.01"},
                        {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                    ],
                }
            ]
        }
        provider = BinanceProvider()

        result = provider.place_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.00009,
        )

        assert result is not None
        sent_qty = mock_client.create_order.call_args.kwargs["quantity"]
        assert isinstance(sent_qty, str)
        assert "e" not in sent_qty.lower()
        assert sent_qty == "0.00009"

    @pytest.mark.fast
    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_place_stop_loss_order_small_quantity_not_scientific_notation(
        self, mock_config, mock_client_class
    ):
        """place_stop_loss_order formats a sub-1e-4 quantity as fixed-point."""
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.create_order.return_value = {"orderId": "sl1"}
        provider = BinanceProvider()
        provider.get_symbol_info = Mock(
            return_value={"step_size": 0.00001, "tick_size": 0.01, "base_asset": "BTC"}
        )
        provider.get_balance = Mock(return_value=Mock(free=1.0))

        result = provider.place_stop_loss_order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            quantity=0.00009,
            stop_price=50000.0,
        )

        assert result == "sl1"
        sent_qty = mock_client.create_order.call_args.kwargs["quantity"]
        assert isinstance(sent_qty, str)
        assert "e" not in sent_qty.lower()
        assert sent_qty == "0.00009"


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
@patch("src.data_providers.binance_provider.BINANCE_AVAILABLE", True)
class TestStopLossPriceTickPrecision:
    """The stopPrice/price sent to Binance must carry no more decimals than PRICE_FILTER allows.

    `round(price / tick) * tick` in float math leaves artifacts (e.g.
    round(1648.82 / 0.01) * 0.01 == 1648.8200000000001). Sent verbatim, that
    over-precise value is rejected with code -1111 ("Parameter 'price' has too much
    precision"), which in prod made every stop-loss placement fail and forced an
    emergency close of the just-opened position. The quantize step must clamp both the
    stopPrice and the derived limit price to the tick's decimal count. Without the fix
    these assertions FAIL (the raw float exponent is far below the tick precision); with
    it the exponent never goes below what the tick allows. Mirror of the quantity 51077
    guard in TestStopLossQuantityStepPrecision, for the price side (#698).
    """

    @staticmethod
    def _make_provider(mock_config, mock_client_class, tick_size, base_asset="ETH"):
        """Build a provider whose symbol_info reports `tick_size` and has free balance."""
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.create_order.return_value = {"orderId": "slp"}
        provider = BinanceProvider()
        provider.get_symbol_info = Mock(
            return_value={
                "step_size": 0.0001,
                "tick_size": tick_size,
                "base_asset": base_asset,
            }
        )
        # Free balance comfortably covers the quantity so the SELL cap never trims it.
        provider.get_balance = Mock(return_value=Mock(free=1_000_000.0))
        return provider, mock_client

    @staticmethod
    def _tick_decimals(tick_size):
        """Number of decimal places the tick size implies (e.g. 0.01 -> 2, 1.0 -> 1)."""
        return max(0, -Decimal(str(tick_size)).as_tuple().exponent)

    @pytest.mark.fast
    @pytest.mark.parametrize(
        ("tick_size", "stop_price"),
        [
            # tick -> a price whose round(price/tick)*tick float round-trip is over-precise.
            (0.01, 1648.82),  # reproduces the live -1111 case; raw exp far below -2
            (0.1, 1648.8),
            (0.01, 2100.07),
            (1.0, 2100.0),  # integer tick: must stay whole, no artifact introduced
        ],
    )
    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_buy_branch_prices_quantized_to_tick(
        self, mock_config, mock_client_class, tick_size, stop_price
    ):
        """BUY branch: stopPrice and the derived limit price carry no excess decimals."""
        provider, mock_client = self._make_provider(mock_config, mock_client_class, tick_size)

        result = provider.place_stop_loss_order(
            symbol="ETHUSDT", side=OrderSide.BUY, quantity=0.05, stop_price=stop_price
        )

        assert result == "slp"
        tick_decimals = self._tick_decimals(tick_size)
        sent_stop = mock_client.create_order.call_args.kwargs["stopPrice"]
        sent_price = mock_client.create_order.call_args.kwargs["price"]
        # Sent as strings; neither may have more decimals than the tick implies.
        assert Decimal(sent_stop).as_tuple().exponent >= -tick_decimals
        assert Decimal(sent_price).as_tuple().exponent >= -tick_decimals
        # stopPrice preserves the intended level (within one tick).
        assert float(sent_stop) == pytest.approx(stop_price, abs=tick_size)

    @pytest.mark.fast
    @pytest.mark.parametrize(
        ("tick_size", "stop_price"),
        [
            (0.01, 1899.93),
            (0.1, 1899.9),
            (0.01, 1648.82),
            (1.0, 1900.0),
        ],
    )
    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_sell_branch_prices_quantized_to_tick(
        self, mock_config, mock_client_class, tick_size, stop_price
    ):
        """SELL branch: stopPrice and the derived limit price carry no excess decimals."""
        provider, mock_client = self._make_provider(mock_config, mock_client_class, tick_size)

        result = provider.place_stop_loss_order(
            symbol="ETHUSDT", side=OrderSide.SELL, quantity=0.05, stop_price=stop_price
        )

        assert result == "slp"
        tick_decimals = self._tick_decimals(tick_size)
        sent_stop = mock_client.create_order.call_args.kwargs["stopPrice"]
        sent_price = mock_client.create_order.call_args.kwargs["price"]
        assert Decimal(sent_stop).as_tuple().exponent >= -tick_decimals
        assert Decimal(sent_price).as_tuple().exponent >= -tick_decimals
        assert float(sent_stop) == pytest.approx(stop_price, abs=tick_size)

    @pytest.mark.fast
    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_reproduces_1111_case_exponent_within_two_decimals(
        self, mock_config, mock_client_class
    ):
        """Explicit -1111 reproduction: tick 0.01, stop 1648.82 -> exponent >= -2."""
        provider, mock_client = self._make_provider(mock_config, mock_client_class, 0.01)

        result = provider.place_stop_loss_order(
            symbol="ETHUSDT", side=OrderSide.BUY, quantity=0.05, stop_price=1648.82
        )

        assert result == "slp"
        sent_stop = mock_client.create_order.call_args.kwargs["stopPrice"]
        assert Decimal(sent_stop).as_tuple().exponent >= -2
        assert float(sent_stop) == pytest.approx(1648.82, abs=1e-9)


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
@patch("src.data_providers.binance_provider.BINANCE_AVAILABLE", True)
class TestOrphanedBorrowExchangeHelpers:
    """repay_margin_loan / get_margin_account_asset / has_open_orders for the sweep."""

    @staticmethod
    def _provider(mock_config, mock_client_class):
        cfg = Mock()
        cfg.get_required.return_value = "fake_key"
        mock_config.return_value = cfg
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        provider = BinanceProvider()
        provider._use_margin = True
        provider._client = mock_client
        return provider, mock_client

    @pytest.mark.fast
    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_repay_uses_modern_borrow_repay_endpoint(self, mock_config, mock_client_class):
        provider, client = self._provider(mock_config, mock_client_class)
        ok = provider.repay_margin_loan("ETH", Decimal("0.00282625"))
        assert ok is True
        client.margin_borrow_repay.assert_called_once_with(
            asset="ETH", amount="0.00282625", type="REPAY", isIsolated="FALSE"
        )
        # The deprecated /sapi/v1/margin/repay endpoint must NOT be used.
        client.repay_margin_loan.assert_not_called()

    @pytest.mark.fast
    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_repay_returns_false_on_error(self, mock_config, mock_client_class):
        provider, client = self._provider(mock_config, mock_client_class)
        client.margin_borrow_repay.side_effect = Exception("APIError(code=-3015) exceeds liability")
        assert provider.repay_margin_loan("ETH", Decimal("0.003")) is False

    @pytest.mark.fast
    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_repay_noop_when_not_margin_or_nonpositive(self, mock_config, mock_client_class):
        provider, client = self._provider(mock_config, mock_client_class)
        provider._use_margin = False
        assert provider.repay_margin_loan("ETH", Decimal("0.003")) is False
        provider._use_margin = True
        assert provider.repay_margin_loan("ETH", Decimal("0")) is False
        client.margin_borrow_repay.assert_not_called()

    @pytest.mark.fast
    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_has_open_orders_fail_closed_on_error(self, mock_config, mock_client_class):
        provider, _ = self._provider(mock_config, mock_client_class)
        with patch.object(provider, "_call_get_open_orders", side_effect=Exception("boom")):
            assert provider.has_open_orders("ETHUSDT") is None  # fail-closed, not []

    @pytest.mark.fast
    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_has_open_orders_true_and_false(self, mock_config, mock_client_class):
        provider, _ = self._provider(mock_config, mock_client_class)
        with patch.object(provider, "_call_get_open_orders", return_value=[{"orderId": 1}]):
            assert provider.has_open_orders("ETHUSDT") is True
        with patch.object(provider, "_call_get_open_orders", return_value=[]):
            assert provider.has_open_orders("ETHUSDT") is False

    @pytest.mark.fast
    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_get_margin_account_asset_returns_raw_strings(self, mock_config, mock_client_class):
        provider, _ = self._provider(mock_config, mock_client_class)
        account = {
            "balances": [
                {
                    "asset": "ETH",
                    "free": "0.0029",
                    "locked": "0",
                    "borrowed": "0.00282625",
                    "interest": "0.0000001",
                    "netAsset": "0.00007",
                }
            ]
        }
        with patch.object(provider, "_call_get_account", return_value=account):
            snap = provider.get_margin_account_asset("ETH")
        assert snap is not None
        assert snap["borrowed"] == "0.00282625"
        assert snap["free"] == "0.0029"
        assert snap["interest"] == "0.0000001"
        assert snap["netAsset"] == "0.00007"


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
@patch("src.data_providers.binance_provider.BINANCE_AVAILABLE", True)
class TestGetOrderIdRouting:
    """Tests for get_order() routing between orderId and origClientOrderId."""

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_numeric_order_id_uses_order_id_param(self, mock_config, mock_client_class):
        """Verify numeric order IDs use Binance orderId parameter."""
        # Arrange
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj

        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_order.return_value = {
            "orderId": 12345,
            "symbol": "BTCUSDT",
            "status": "FILLED",
            "side": "BUY",
            "type": "MARKET",
            "origQty": "1.0",
            "executedQty": "1.0",
            "price": "50000.0",
            "time": 1640995200000,
        }

        provider = BinanceProvider()

        # Act
        provider.get_order("12345", "BTCUSDT")

        # Assert - uses orderId, not origClientOrderId
        mock_client.get_order.assert_called_once_with(symbol="BTCUSDT", orderId="12345")

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_alphanumeric_order_id_delegates_to_get_order_by_client_id(
        self, mock_config, mock_client_class
    ):
        """Verify alphanumeric order IDs delegate to get_order_by_client_id."""
        # Arrange
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj

        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_order.return_value = {
            "orderId": 12345,
            "symbol": "BTCUSDT",
            "status": "FILLED",
            "side": "BUY",
            "type": "MARKET",
            "origQty": "1.0",
            "executedQty": "1.0",
            "price": "50000.0",
            "time": 1640995200000,
        }

        provider = BinanceProvider()

        # Act
        provider.get_order("atb_19d360981ab_3a4b0d5a", "BTCUSDT")

        # Assert - delegates to get_order_by_client_id which uses origClientOrderId
        mock_client.get_order.assert_called_once_with(
            symbol="BTCUSDT", origClientOrderId="atb_19d360981ab_3a4b0d5a"
        )

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_underscore_order_id_delegates_to_get_order_by_client_id(
        self, mock_config, mock_client_class
    ):
        """Verify order IDs with underscores delegate to get_order_by_client_id."""
        # Arrange
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj

        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_order.return_value = {
            "orderId": 99999,
            "symbol": "ETHUSDT",
            "status": "NEW",
            "side": "SELL",
            "type": "LIMIT",
            "origQty": "2.0",
            "executedQty": "0.0",
            "price": "3000.0",
            "time": 1640995200000,
        }

        provider = BinanceProvider()

        # Act
        provider.get_order("client_order_123", "ETHUSDT")

        # Assert
        mock_client.get_order.assert_called_once_with(
            symbol="ETHUSDT", origClientOrderId="client_order_123"
        )

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_get_order_returns_none_on_api_error(self, mock_config, mock_client_class):
        """Verify get_order returns None when Binance API raises an error."""
        # Arrange
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj

        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_order.side_effect = Exception("API error")

        provider = BinanceProvider()

        # Act
        result = provider.get_order("atb_bad_id", "BTCUSDT")

        # Assert
        assert result is None


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
@patch("src.data_providers.binance_provider.BINANCE_AVAILABLE", True)
class TestGetOrderChecked:
    """Fail-closed order lookup (#713).

    get_order_checked must return None ONLY when Binance confirms the order
    does not exist (-2013); any unconfirmed lookup raises OrderLookupError so
    safety logic (reconciler stop-loss re-placement) never mistakes a transient
    API failure for a missing order.
    """

    ORDER_PAYLOAD = {
        "orderId": 12345,
        "symbol": "BTCUSDT",
        "status": "NEW",
        "side": "SELL",
        "type": "STOP_LOSS_LIMIT",
        "origQty": "1.0",
        "executedQty": "0.0",
        "price": "50000.0",
        "time": 1640995200000,
        "updateTime": 1640995200000,
    }

    @staticmethod
    def _make_provider(mock_config, mock_client_class):
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        return BinanceProvider(), mock_client

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_returns_order_when_found(self, mock_config, mock_client_class):
        provider, mock_client = self._make_provider(mock_config, mock_client_class)
        mock_client.get_order.return_value = dict(self.ORDER_PAYLOAD)

        order = provider.get_order_checked("12345", "BTCUSDT")

        assert order is not None
        assert order.order_id == "12345"
        mock_client.get_order.assert_called_once_with(symbol="BTCUSDT", orderId="12345")

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_returns_none_only_on_confirmed_not_found(self, mock_config, mock_client_class):
        from binance.exceptions import BinanceAPIException

        provider, mock_client = self._make_provider(mock_config, mock_client_class)
        mock_client.get_order.side_effect = BinanceAPIException(
            Mock(), 400, '{"code": -2013, "msg": "Order does not exist."}'
        )

        assert provider.get_order_checked("12345", "BTCUSDT") is None

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_raises_on_api_error(self, mock_config, mock_client_class):
        from binance.exceptions import BinanceAPIException

        provider, mock_client = self._make_provider(mock_config, mock_client_class)
        mock_client.get_order.side_effect = BinanceAPIException(
            Mock(), 429, '{"code": -1003, "msg": "Too many requests."}'
        )

        with pytest.raises(OrderLookupError):
            provider.get_order_checked("12345", "BTCUSDT")

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_raises_on_network_error(self, mock_config, mock_client_class):
        provider, mock_client = self._make_provider(mock_config, mock_client_class)
        mock_client.get_order.side_effect = ConnectionError("connection reset")

        with pytest.raises(OrderLookupError):
            provider.get_order_checked("12345", "BTCUSDT")

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_raises_when_client_unavailable(self, mock_config, mock_client_class):
        provider, _ = self._make_provider(mock_config, mock_client_class)
        provider._client = None

        with pytest.raises(OrderLookupError):
            provider.get_order_checked("12345", "BTCUSDT")

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_alphanumeric_id_routes_to_client_order_id(self, mock_config, mock_client_class):
        provider, mock_client = self._make_provider(mock_config, mock_client_class)
        mock_client.get_order.return_value = dict(self.ORDER_PAYLOAD)

        order = provider.get_order_checked("atb_19d360981ab_3a4b0d5a", "BTCUSDT")

        assert order is not None
        mock_client.get_order.assert_called_once_with(
            symbol="BTCUSDT", origClientOrderId="atb_19d360981ab_3a4b0d5a"
        )


# ========================================
# Margin Trading Tests
# ========================================


def _make_config_mock(overrides: dict | None = None):
    """Create a config mock with sensible defaults and dict-backed .get().

    Args:
        overrides: Key-value pairs to override defaults (e.g. BINANCE_ACCOUNT_TYPE).

    Returns:
        Configured Mock object for get_config().
    """
    defaults = {
        "BINANCE_ACCOUNT_TYPE": "spot",
        "TRADING_MODE": "paper",
        "ENV": "test",
    }
    if overrides:
        defaults.update(overrides)

    mock_config_obj = Mock()
    mock_config_obj.get.side_effect = lambda key, default=None: defaults.get(key, default)
    mock_config_obj.get_required.return_value = "fake_key"
    mock_config_obj.get_float.return_value = 60.0
    return mock_config_obj


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
class TestMarginFlag:
    """Tests for margin flag initialization from config."""

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_flag_from_env_margin(self, mock_config, mock_client_class):
        """Verify _use_margin=True when BINANCE_ACCOUNT_TYPE=margin."""
        mock_config.return_value = _make_config_mock({"BINANCE_ACCOUNT_TYPE": "margin"})
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_margin_account.return_value = {
            "tradeEnabled": True,
            "borrowEnabled": True,
            "marginLevel": "2.5",
            "userAssets": [],
        }

        provider = BinanceProvider()
        assert provider._use_margin is True
        assert provider._is_live is False

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_flag_from_env_spot(self, mock_config, mock_client_class):
        """Verify _use_margin=False when BINANCE_ACCOUNT_TYPE=spot."""
        mock_config.return_value = _make_config_mock({"BINANCE_ACCOUNT_TYPE": "spot"})
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        provider = BinanceProvider()
        assert provider._use_margin is False

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_is_live_flag(self, mock_config, mock_client_class):
        """Verify _is_live=True when TRADING_MODE=live."""
        mock_config.return_value = _make_config_mock(
            {
                "BINANCE_ACCOUNT_TYPE": "spot",
                "TRADING_MODE": "live",
            }
        )
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        provider = BinanceProvider()
        assert provider._is_live is True

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_symbol_verified_cache_initialized(self, mock_config, mock_client_class):
        """Verify _margin_symbol_verified set is initialized empty."""
        mock_config.return_value = _make_config_mock()
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        provider = BinanceProvider()
        assert provider._margin_symbol_verified == set()


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
class TestMarginFailFast:
    """Tests for live+margin fail-fast on client init failure."""

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_live_mode_no_offline_fallback(self, mock_config, mock_client_class):
        """Live+margin raises RuntimeError instead of falling back to offline stub."""
        mock_config.return_value = _make_config_mock(
            {
                "BINANCE_ACCOUNT_TYPE": "margin",
                "TRADING_MODE": "live",
            }
        )
        mock_client_class.side_effect = Exception("Connection refused")

        with pytest.raises(RuntimeError, match="FATAL.*live margin mode"):
            BinanceProvider()

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_paper_mode_allows_offline_fallback(self, mock_config, mock_client_class):
        """Paper+margin falls back to offline stub (no RuntimeError)."""
        mock_config.return_value = _make_config_mock(
            {
                "BINANCE_ACCOUNT_TYPE": "margin",
                "TRADING_MODE": "paper",
            }
        )
        mock_client_class.side_effect = Exception("Connection refused")

        provider = BinanceProvider()
        assert provider._client is not None  # offline stub

    @patch("src.data_providers.binance_provider.BINANCE_AVAILABLE", False)
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_live_no_sdk_fails_fast(self, mock_config):
        """Live+margin raises RuntimeError when SDK is not installed."""
        mock_config.return_value = _make_config_mock(
            {
                "BINANCE_ACCOUNT_TYPE": "margin",
                "TRADING_MODE": "live",
            }
        )

        with pytest.raises(RuntimeError, match="Binance library not available"):
            BinanceProvider()


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
class TestMarginStartupChecks:
    """Tests for margin account verification at startup."""

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_startup_checks_pass(self, mock_config, mock_client_class):
        """Startup passes when tradeEnabled and borrowEnabled are True."""
        mock_config.return_value = _make_config_mock({"BINANCE_ACCOUNT_TYPE": "margin"})
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_margin_account.return_value = {
            "tradeEnabled": True,
            "borrowEnabled": True,
            "marginLevel": "2.5",
            "userAssets": [],
        }

        provider = BinanceProvider()
        assert provider._use_margin is True

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_startup_trade_disabled_raises(self, mock_config, mock_client_class):
        """RuntimeError raised when tradeEnabled=False."""
        mock_config.return_value = _make_config_mock({"BINANCE_ACCOUNT_TYPE": "margin"})
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_margin_account.return_value = {
            "tradeEnabled": False,
            "borrowEnabled": True,
            "marginLevel": "2.5",
        }

        with pytest.raises(RuntimeError, match="tradeEnabled=False"):
            BinanceProvider()

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_startup_borrow_disabled_raises(self, mock_config, mock_client_class):
        """RuntimeError raised when borrowEnabled=False."""
        mock_config.return_value = _make_config_mock({"BINANCE_ACCOUNT_TYPE": "margin"})
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_margin_account.return_value = {
            "tradeEnabled": True,
            "borrowEnabled": False,
            "marginLevel": "2.5",
        }

        with pytest.raises(RuntimeError, match="borrowEnabled=False"):
            BinanceProvider()

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_startup_api_error_live_raises(self, mock_config, mock_client_class):
        """API error during margin verification in live mode raises RuntimeError."""
        mock_config.return_value = _make_config_mock(
            {
                "BINANCE_ACCOUNT_TYPE": "margin",
                "TRADING_MODE": "live",
            }
        )
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_margin_account.side_effect = Exception("API timeout")

        with pytest.raises(RuntimeError, match="Failed to verify margin account"):
            BinanceProvider()

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_startup_api_error_paper_warns(self, mock_config, mock_client_class):
        """API error during margin verification in paper mode logs warning, no raise."""
        mock_config.return_value = _make_config_mock(
            {
                "BINANCE_ACCOUNT_TYPE": "margin",
                "TRADING_MODE": "paper",
            }
        )
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_margin_account.side_effect = Exception("API timeout")

        # Should not raise — just warns
        provider = BinanceProvider()
        assert provider._use_margin is True


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
class TestMarginBaseAssetGuard:
    """Tests for the non-USDT base asset startup guard."""

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_warns_non_usdt_holdings_in_live_mode(self, mock_config, mock_client_class):
        """Live margin mode warns (not raises) for non-USDT assets — could be a recovering long."""
        mock_config.return_value = _make_config_mock(
            {
                "BINANCE_ACCOUNT_TYPE": "margin",
                "TRADING_MODE": "live",
            }
        )
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_margin_account.return_value = {
            "tradeEnabled": True,
            "borrowEnabled": True,
            "marginLevel": "2.5",
            "userAssets": [
                {"asset": "USDT", "free": "100", "locked": "0", "netAsset": "100"},
                {"asset": "ETH", "free": "0.05", "locked": "0", "netAsset": "0.05"},
            ],
        }
        mock_client.get_symbol_ticker.return_value = {"price": "2000"}

        # Should warn but NOT raise — could be a recovering long position.
        # Provider init runs before startup reconciliation, so blocking
        # prevents position recovery.
        provider = BinanceProvider()
        assert provider._use_margin is True

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_allows_dust_holdings(self, mock_config, mock_client_class):
        """Dust amounts (< $1) are ignored."""
        mock_config.return_value = _make_config_mock(
            {
                "BINANCE_ACCOUNT_TYPE": "margin",
                "TRADING_MODE": "live",
            }
        )
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_margin_account.return_value = {
            "tradeEnabled": True,
            "borrowEnabled": True,
            "marginLevel": "2.5",
            "userAssets": [
                {"asset": "USDT", "free": "100", "locked": "0", "netAsset": "100"},
                {"asset": "ETH", "free": "0.0001", "locked": "0", "netAsset": "0.0001"},
            ],
        }
        mock_client.get_symbol_ticker.return_value = {"price": "2000"}

        # $0.20 worth of ETH — below $1 threshold, should not raise
        provider = BinanceProvider()
        assert provider._use_margin is True

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_warns_in_paper_mode(self, mock_config, mock_client_class):
        """Paper mode warns but doesn't raise for non-USDT holdings."""
        mock_config.return_value = _make_config_mock(
            {
                "BINANCE_ACCOUNT_TYPE": "margin",
                "TRADING_MODE": "paper",
            }
        )
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_margin_account.return_value = {
            "tradeEnabled": True,
            "borrowEnabled": True,
            "marginLevel": "2.5",
            "userAssets": [
                {"asset": "USDT", "free": "100", "locked": "0", "netAsset": "100"},
                {"asset": "ETH", "free": "0.05", "locked": "0", "netAsset": "0.05"},
            ],
        }
        mock_client.get_symbol_ticker.return_value = {"price": "2000"}

        # Paper mode — should warn, not raise
        provider = BinanceProvider()
        assert provider._use_margin is True


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
class TestMarginDispatch:
    """Tests for margin/spot dispatch routing."""

    def _make_provider(self, mock_config, mock_client_class, use_margin=True):
        """Helper to create a provider with margin config."""
        account_type = "margin" if use_margin else "spot"
        mock_config.return_value = _make_config_mock({"BINANCE_ACCOUNT_TYPE": account_type})
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        if use_margin:
            mock_client.get_margin_account.return_value = {
                "tradeEnabled": True,
                "borrowEnabled": True,
                "marginLevel": "2.5",
                "userAssets": [],
            }
            mock_client.get_margin_symbol.return_value = {
                "isMarginTrade": True,
                "isBuyAllowed": True,
                "isSellAllowed": True,
            }
        provider = BinanceProvider()
        return provider, mock_client

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_dispatch_routes_to_margin_methods(self, mock_config, mock_client_class):
        """Dispatch methods call margin client methods when _use_margin=True."""
        provider, mock_client = self._make_provider(mock_config, mock_client_class, use_margin=True)

        # _call_get_account
        mock_client.get_margin_account.return_value = {
            "tradeEnabled": True,
            "borrowEnabled": True,
            "userAssets": [{"asset": "BTC", "free": "1.0", "locked": "0.0"}],
        }
        provider._call_get_account()
        mock_client.get_margin_account.assert_called()

        # _call_create_order
        mock_client.create_margin_order.return_value = {"orderId": "123"}
        provider._call_create_order(symbol="BTCUSDT", side="BUY", type="MARKET", quantity=0.1)
        mock_client.create_margin_order.assert_called()

        # _call_get_order
        provider._call_get_order(symbol="BTCUSDT", orderId="123")
        mock_client.get_margin_order.assert_called()

        # _call_get_open_orders
        mock_client.get_open_margin_orders.return_value = []
        provider._call_get_open_orders(symbol="BTCUSDT")
        mock_client.get_open_margin_orders.assert_called()

        # _call_get_my_trades
        mock_client.get_margin_trades.return_value = []
        provider._call_get_my_trades(symbol="BTCUSDT")
        mock_client.get_margin_trades.assert_called()

        # _call_cancel_order
        provider._call_cancel_order(symbol="BTCUSDT", orderId="123")
        mock_client.cancel_margin_order.assert_called()

        # _call_get_all_orders
        mock_client.get_all_margin_orders.return_value = []
        provider._call_get_all_orders(symbol="BTCUSDT")
        mock_client.get_all_margin_orders.assert_called()

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_spot_dispatch_routes_to_spot_methods(self, mock_config, mock_client_class):
        """Dispatch methods call spot client methods when _use_margin=False."""
        provider, mock_client = self._make_provider(
            mock_config, mock_client_class, use_margin=False
        )

        # _call_get_account
        mock_client.get_account.return_value = {"balances": [], "canTrade": True}
        provider._call_get_account()
        mock_client.get_account.assert_called()

        # _call_create_order
        mock_client.create_order.return_value = {"orderId": "123"}
        provider._call_create_order(symbol="BTCUSDT", side="BUY", type="MARKET", quantity=0.1)
        mock_client.create_order.assert_called()

        # _call_get_order
        provider._call_get_order(symbol="BTCUSDT", orderId="123")
        mock_client.get_order.assert_called()

        # _call_get_open_orders
        mock_client.get_open_orders.return_value = []
        provider._call_get_open_orders(symbol="BTCUSDT")
        mock_client.get_open_orders.assert_called()

        # _call_get_my_trades
        mock_client.get_my_trades.return_value = []
        provider._call_get_my_trades(symbol="BTCUSDT")
        mock_client.get_my_trades.assert_called()

        # _call_cancel_order
        provider._call_cancel_order(symbol="BTCUSDT", orderId="123")
        mock_client.cancel_order.assert_called()

        # _call_get_all_orders
        mock_client.get_all_orders.return_value = []
        provider._call_get_all_orders(symbol="BTCUSDT")
        mock_client.get_all_orders.assert_called()

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_dispatch_adds_isIsolated_false(self, mock_config, mock_client_class):
        """Margin dispatch injects isIsolated=FALSE for cross-margin mode."""
        provider, mock_client = self._make_provider(mock_config, mock_client_class, use_margin=True)

        mock_client.get_open_margin_orders.return_value = []
        provider._call_get_open_orders(symbol="BTCUSDT")
        call_kwargs = mock_client.get_open_margin_orders.call_args.kwargs
        assert call_kwargs["isIsolated"] == "FALSE"

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_spot_dispatch_strips_sideEffectType(self, mock_config, mock_client_class):
        """Spot dispatch removes sideEffectType param to avoid Binance error."""
        provider, mock_client = self._make_provider(
            mock_config, mock_client_class, use_margin=False
        )

        mock_client.create_order.return_value = {"orderId": "123"}
        provider._call_create_order(
            symbol="BTCUSDT",
            side="BUY",
            type="MARKET",
            quantity=0.1,
            sideEffectType="MARGIN_BUY",
        )
        call_kwargs = mock_client.create_order.call_args.kwargs
        assert "sideEffectType" not in call_kwargs


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
class TestMarginBalanceNormalization:
    """Tests for margin account balance normalization."""

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_balance_normalization(self, mock_config, mock_client_class):
        """_call_get_account normalizes userAssets to balances format."""
        mock_config.return_value = _make_config_mock({"BINANCE_ACCOUNT_TYPE": "margin"})
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_margin_account.return_value = {
            "tradeEnabled": True,
            "borrowEnabled": True,
            "marginLevel": "2.5",
            "userAssets": [
                {
                    "asset": "BTC",
                    "free": "1.0",
                    "locked": "0.5",
                    "borrowed": "0.2",
                    "interest": "0.001",
                    "netAsset": "1.299",
                },
                {
                    "asset": "USDT",
                    "free": "10000",
                    "locked": "0",
                    "borrowed": "0",
                    "interest": "0",
                    "netAsset": "10000",
                },
            ],
        }
        mock_client.get_margin_symbol.return_value = {
            "isMarginTrade": True,
            "isBuyAllowed": True,
            "isSellAllowed": True,
        }

        provider = BinanceProvider()
        result = provider._call_get_account()

        assert "balances" in result
        assert len(result["balances"]) == 2
        btc_bal = result["balances"][0]
        assert btc_bal["asset"] == "BTC"
        assert btc_bal["free"] == "1.0"
        assert btc_bal["locked"] == "0.5"
        assert btc_bal["borrowed"] == "0.2"
        assert btc_bal["interest"] == "0.001"
        assert btc_bal["netAsset"] == "1.299"
        assert result["canTrade"] is True

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_spot_balance_passthrough(self, mock_config, mock_client_class):
        """_call_get_account returns spot account data unchanged."""
        mock_config.return_value = _make_config_mock({"BINANCE_ACCOUNT_TYPE": "spot"})
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        spot_data = {"balances": [{"asset": "BTC", "free": "1.0", "locked": "0"}], "canTrade": True}
        mock_client.get_account.return_value = spot_data

        provider = BinanceProvider()
        result = provider._call_get_account()
        assert result == spot_data


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
class TestMarginSymbolValidation:
    """Tests for lazy margin symbol validation."""

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_symbol_validation_on_first_order(self, mock_config, mock_client_class):
        """Symbol is validated on first margin order, cached for subsequent ones."""
        mock_config.return_value = _make_config_mock({"BINANCE_ACCOUNT_TYPE": "margin"})
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_margin_account.return_value = {
            "tradeEnabled": True,
            "borrowEnabled": True,
            "marginLevel": "2.5",
            "userAssets": [],
        }
        mock_client.get_margin_symbol.return_value = {
            "isMarginTrade": True,
            "isBuyAllowed": True,
            "isSellAllowed": True,
        }
        mock_client.create_margin_order.return_value = {"orderId": "123"}

        provider = BinanceProvider()
        # Reset call count after __init__ verification
        mock_client.get_margin_symbol.reset_mock()

        # First BUY order triggers validation
        provider._call_create_order(symbol="BTCUSDT", side="BUY", type="MARKET", quantity=0.1)
        assert mock_client.get_margin_symbol.call_count == 1

        # Second BUY order with same symbol+side skips validation (cached)
        provider._call_create_order(symbol="BTCUSDT", side="BUY", type="MARKET", quantity=0.1)
        assert mock_client.get_margin_symbol.call_count == 1  # Still 1

        # SELL order for same symbol triggers new validation (different side)
        provider._call_create_order(symbol="BTCUSDT", side="SELL", type="MARKET", quantity=0.1)
        assert mock_client.get_margin_symbol.call_count == 2

        # Second SELL skips (cached)
        provider._call_create_order(symbol="BTCUSDT", side="SELL", type="MARKET", quantity=0.1)
        assert mock_client.get_margin_symbol.call_count == 2  # Still 2

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_symbol_not_supported_raises(self, mock_config, mock_client_class):
        """RuntimeError raised if symbol doesn't support margin trading."""
        mock_config.return_value = _make_config_mock({"BINANCE_ACCOUNT_TYPE": "margin"})
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_margin_account.return_value = {
            "tradeEnabled": True,
            "borrowEnabled": True,
            "marginLevel": "2.5",
            "userAssets": [],
        }
        # Return unsupported on the order-time check
        mock_client.get_margin_symbol.side_effect = [
            # Called during _call_create_order for symbol validation
            {"isMarginTrade": False, "isBuyAllowed": True, "isSellAllowed": True},
        ]
        mock_client.create_margin_order.return_value = {"orderId": "123"}

        provider = BinanceProvider()

        with pytest.raises(ValueError, match="does not support margin trading"):
            provider._call_create_order(symbol="BADUSDT", side="BUY", type="MARKET", quantity=0.1)


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
class TestMarginSideEffectIntegration:
    """Verify side_effect_type flows through place_order/place_stop_loss_order."""

    # Minimal exchange_info symbol data that satisfies get_symbol_info
    _ETHUSDT_SYMBOL = {
        "symbol": "ETHUSDT",
        "baseAsset": "ETH",
        "quoteAsset": "USDT",
        "status": "TRADING",
        "filters": [
            {"filterType": "LOT_SIZE", "minQty": "0.001", "stepSize": "0.001"},
            {"filterType": "PRICE_FILTER", "minPrice": "0.01", "tickSize": "0.01"},
            {"filterType": "MIN_NOTIONAL", "minNotional": "10"},
        ],
    }

    def _make_margin_provider(self, mock_config, mock_client_class):
        """Create a margin-mode provider with all mocks configured."""
        mock_config.return_value = _make_config_mock({"BINANCE_ACCOUNT_TYPE": "margin"})
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_margin_account.return_value = {
            "tradeEnabled": True,
            "borrowEnabled": True,
            "marginLevel": "2.5",
            "userAssets": [],
        }
        mock_client.get_margin_symbol.return_value = {
            "isMarginTrade": True,
            "isBuyAllowed": True,
            "isSellAllowed": True,
        }
        mock_client.get_exchange_info.return_value = {
            "symbols": [self._ETHUSDT_SYMBOL],
        }
        mock_client.create_margin_order.return_value = {
            "orderId": "99",
            "status": "FILLED",
            "origQty": "0.1",
            "executedQty": "0.1",
            "cummulativeQuoteQty": "200.0",
            "fills": [],
        }
        return BinanceProvider(), mock_client

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_place_order_passes_side_effect_type(self, mock_config, mock_client_class):
        """place_order() injects sideEffectType into margin create_margin_order call."""
        from src.data_providers.exchange_interface import OrderType

        provider, mock_client = self._make_margin_provider(mock_config, mock_client_class)

        provider.place_order(
            symbol="ETHUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=0.05,
            side_effect_type="MARGIN_BUY",
        )

        call_kwargs = mock_client.create_margin_order.call_args.kwargs
        assert call_kwargs["sideEffectType"] == "MARGIN_BUY"
        assert call_kwargs["isIsolated"] == "FALSE"

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_place_stop_loss_passes_side_effect_type(self, mock_config, mock_client_class):
        """place_stop_loss_order() injects sideEffectType into margin order."""
        provider, mock_client = self._make_margin_provider(mock_config, mock_client_class)

        provider.place_stop_loss_order(
            symbol="ETHUSDT",
            side=OrderSide.SELL,
            quantity=0.05,
            stop_price=1800.0,
            side_effect_type="AUTO_REPAY",
        )

        call_kwargs = mock_client.create_margin_order.call_args.kwargs
        assert call_kwargs["sideEffectType"] == "AUTO_REPAY"
        assert call_kwargs["isIsolated"] == "FALSE"

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_place_order_no_side_effect_in_spot_mode(self, mock_config, mock_client_class):
        """place_order() does not inject sideEffectType in spot mode."""
        from src.data_providers.exchange_interface import OrderType

        mock_config.return_value = _make_config_mock({"BINANCE_ACCOUNT_TYPE": "spot"})
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_exchange_info.return_value = {
            "symbols": [self._ETHUSDT_SYMBOL],
        }
        mock_client.create_order.return_value = {
            "orderId": "99",
            "status": "FILLED",
            "origQty": "0.1",
            "executedQty": "0.1",
            "cummulativeQuoteQty": "200.0",
            "fills": [],
        }

        provider = BinanceProvider()
        provider.place_order(
            symbol="ETHUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=0.05,
            side_effect_type="MARGIN_BUY",
        )

        call_kwargs = mock_client.create_order.call_args.kwargs
        assert "sideEffectType" not in call_kwargs


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
class TestMarginErrorCodes:
    """Verify margin-specific Binance error codes are treated as definitive rejects."""

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_margin_insufficient_balance_raises(self, mock_config, mock_client_class):
        """Error -3041 (insufficient margin balance) raises ValueError, not None."""
        from binance.exceptions import BinanceAPIException

        from src.data_providers.exchange_interface import OrderType

        mock_config.return_value = _make_config_mock({"BINANCE_ACCOUNT_TYPE": "margin"})
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_margin_account.return_value = {
            "tradeEnabled": True,
            "borrowEnabled": True,
            "marginLevel": "2.5",
            "userAssets": [],
        }
        mock_client.get_margin_symbol.return_value = {
            "isMarginTrade": True,
            "isBuyAllowed": True,
            "isSellAllowed": True,
        }
        mock_client.get_exchange_info.return_value = {
            "symbols": [
                {
                    "symbol": "ETHUSDT",
                    "baseAsset": "ETH",
                    "quoteAsset": "USDT",
                    "status": "TRADING",
                    "filters": [
                        {"filterType": "LOT_SIZE", "minQty": "0.001", "stepSize": "0.001"},
                        {"filterType": "PRICE_FILTER", "minPrice": "0.01", "tickSize": "0.01"},
                        {"filterType": "MIN_NOTIONAL", "minNotional": "10"},
                    ],
                }
            ],
        }

        # Simulate Binance margin error -3041
        error = BinanceAPIException(
            Mock(status_code=400, headers={}),
            400,
            '{"code":-3041,"msg":"Balance is not enough"}',
        )
        error.code = -3041
        error.message = "Balance is not enough"
        mock_client.create_margin_order.side_effect = error

        provider = BinanceProvider()
        with pytest.raises(ValueError, match="Order rejected by exchange.*-3041"):
            provider.place_order(
                symbol="ETHUSDT",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=0.05,
                side_effect_type="MARGIN_BUY",
            )


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
class TestConvertOrderType:
    """Regression tests for #1152: STOP_LOSS_LIMIT/TAKE_PROFIT_LIMIT silently mapped to MARKET.

    `place_stop_loss_order` sends `"type": "STOP_LOSS_LIMIT"` and Binance echoes that same
    string back on every order lookup — not the bare "STOP_LOSS" the old mapping expected.
    The missing key meant `.get(..., OrderType.MARKET)` fell back to MARKET for every
    stop-loss order ever parsed.
    """

    def _make_provider(self):
        with patch("src.data_providers.binance_provider.get_config") as mock_config:
            mock_config_obj = Mock()
            mock_config_obj.get_required.return_value = "fake_key"
            mock_config.return_value = mock_config_obj
            return BinanceProvider()

    def test_stop_loss_limit_maps_to_stop_loss(self):
        from src.data_providers.exchange_interface import OrderType

        provider = self._make_provider()
        assert provider._convert_order_type("STOP_LOSS_LIMIT") == OrderType.STOP_LOSS

    def test_take_profit_limit_maps_to_take_profit(self):
        from src.data_providers.exchange_interface import OrderType

        provider = self._make_provider()
        assert provider._convert_order_type("TAKE_PROFIT_LIMIT") == OrderType.TAKE_PROFIT

    def test_stop_loss_limit_order_parses_with_correct_order_type(self):
        from src.data_providers.exchange_interface import OrderType

        provider = self._make_provider()
        order_data = {
            "orderId": 12345,
            "symbol": "BTCUSDT",
            "side": "SELL",
            "type": "STOP_LOSS_LIMIT",
            "origQty": "0.01",
            "price": "49000.00",
            "stopPrice": "49500.00",
            "status": "NEW",
            "executedQty": "0",
            "time": 1700000000000,
            "updateTime": 1700000000000,
        }

        order = provider._parse_order_data(order_data)

        assert order is not None
        assert order.order_type == OrderType.STOP_LOSS


class TestParseOrderDataStopPrice:
    """#1112: get_open_orders_checked keys resting-stop detection on stop_price
    being non-None -- a misparse here would misclassify every order."""

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_plain_order_stop_price_zero_string_parses_to_none(
        self, mock_config, mock_client_class
    ):
        """Binance sends "0.00000000" (not "0") for a plain order's stopPrice.
        The old `!= "0"` check let this through as a truthy 0.0, indistinguishable
        from a genuine stop resting at price 0 -- misclassifying every plain
        limit/market order as a resting stop."""
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj
        mock_client_class.return_value = Mock()
        provider = BinanceProvider()

        order = provider._parse_order_data(
            {
                "orderId": 1,
                "symbol": "BTCUSDT",
                "side": "SELL",
                "type": "LIMIT",
                "origQty": "0.1",
                "status": "NEW",
                "executedQty": "0",
                "time": 1700000000000,
                "updateTime": 1700000000000,
                "stopPrice": "0.00000000",
            }
        )
        assert order is not None
        assert order.stop_price is None

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_genuine_stop_order_stop_price_parses_correctly(self, mock_config, mock_client_class):
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj
        mock_client_class.return_value = Mock()
        provider = BinanceProvider()

        order = provider._parse_order_data(
            {
                "orderId": 2,
                "symbol": "BTCUSDT",
                "side": "SELL",
                "type": "STOP_LOSS_LIMIT",
                "origQty": "0.1",
                "status": "NEW",
                "executedQty": "0",
                "time": 1700000000000,
                "updateTime": 1700000000000,
                "stopPrice": "48000.00000000",
            }
        )
        assert order is not None
        assert order.stop_price == pytest.approx(48000.0)


class TestGetOpenOrdersCheckedFailClosed:
    """#1112: get_open_orders_checked must not silently drop an unparseable
    order -- if that row was the resting stop, dropping it makes "confirmed
    empty" indistinguishable from "one order we couldn't read might be a stop"."""

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_returns_none_when_any_order_fails_to_parse(self, mock_config, mock_client_class):
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        provider = BinanceProvider()

        good_order = {
            "orderId": 1,
            "symbol": "BTCUSDT",
            "side": "SELL",
            "type": "LIMIT",
            "origQty": "0.1",
            "status": "NEW",
            "executedQty": "0",
            "time": 1700000000000,
            "updateTime": 1700000000000,
        }
        malformed_order = {"orderId": 2}  # missing required fields -> parse fails
        provider._call_get_open_orders = Mock(return_value=[good_order, malformed_order])

        result = provider.get_open_orders_checked("BTCUSDT")
        assert result is None

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_returns_full_list_when_all_orders_parse(self, mock_config, mock_client_class):
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        provider = BinanceProvider()

        good_order = {
            "orderId": 1,
            "symbol": "BTCUSDT",
            "side": "SELL",
            "type": "LIMIT",
            "origQty": "0.1",
            "status": "NEW",
            "executedQty": "0",
            "time": 1700000000000,
            "updateTime": 1700000000000,
        }
        provider._call_get_open_orders = Mock(return_value=[good_order])

        result = provider.get_open_orders_checked("BTCUSDT")
        assert result is not None
        assert len(result) == 1

    @pytest.mark.parametrize("error_code", [-1003, -1015])
    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_reraises_on_rate_limit_ban_instead_of_swallowing_to_none(
        self, mock_config, mock_client_class, error_code
    ):
        """#738 review follow-up: a -1003/-1015 exchange-wide rate-limit ban
        hitting THIS lookup must re-raise (not swallow to a bare None) so
        guard_stop_placement's caller can tell "we're banned" from a generic
        transient lookup failure. Before this fix every exception here --
        ban included -- was swallowed into the same None, which made the
        stop-placement guard's REFUSE indistinguishable from an ordinary
        unconfirmed lookup and left on_rate_limit_ban unreachable from this
        path.
        """
        from binance.exceptions import BinanceAPIException

        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        provider = BinanceProvider()

        ban_error = BinanceAPIException(Mock(status_code=418, headers={}), 418, "")
        ban_error.code = error_code
        provider._call_get_open_orders = Mock(side_effect=ban_error)

        with pytest.raises(BinanceAPIException) as exc_info:
            provider.get_open_orders_checked("BTCUSDT")
        assert exc_info.value.code == error_code

    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_non_ban_exception_still_returns_none(self, mock_config, mock_client_class):
        """Every non-ban exception keeps the existing fail-closed None return --
        only the specific rate-limit ban codes change behavior."""
        mock_config_obj = Mock()
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        provider = BinanceProvider()

        provider._call_get_open_orders = Mock(side_effect=RuntimeError("network blip"))

        result = provider.get_open_orders_checked("BTCUSDT")
        assert result is None
