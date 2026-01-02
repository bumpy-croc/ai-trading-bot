"""Tests for infrastructure.runtime.geo module."""

from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.runtime.geo import (
    _geo_cache,
    clear_geo_cache,
    get_binance_api_endpoint,
    get_country_code,
    is_us_location,
)


class TestGetCountryCode:
    """Tests for get_country_code function."""

    def setup_method(self):
        """Clear geo cache before each test."""
        clear_geo_cache()

    def test_returns_cached_value(self):
        """Test that cached country code is returned."""
        import src.infrastructure.runtime.geo as geo_module

        geo_module._geo_cache = ("US", "https://ipapi.co/country/")
        result = get_country_code()
        assert result == "US"
        geo_module._geo_cache = None

    def test_ipapi_co_success(self):
        """Test successful response from ipapi.co."""
        clear_geo_cache()
        mock_response = MagicMock()
        mock_response.text = "GB"
        mock_response.raise_for_status = MagicMock()

        with patch("src.infrastructure.runtime.geo.requests.get", return_value=mock_response):
            result = get_country_code()
            assert result == "GB"

    def test_ipinfo_io_success(self):
        """Test successful response from ipinfo.io."""
        clear_geo_cache()
        mock_response = MagicMock()
        mock_response.text = "DE"
        mock_response.raise_for_status = MagicMock()

        # First service fails, second succeeds
        def mock_get(url, timeout):
            if "ipapi.co" in url:
                raise Exception("Service unavailable")
            mock_resp = MagicMock()
            mock_resp.text = "DE"
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with patch("src.infrastructure.runtime.geo.requests.get", side_effect=mock_get):
            result = get_country_code()
            assert result == "DE"

    def test_ip_api_json_response(self):
        """Test successful JSON response from ip-api.com."""
        clear_geo_cache()

        def mock_get(url, timeout):
            if "ip-api.com" not in url:
                raise Exception("Service unavailable")
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"countryCode": "fr"}
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with patch("src.infrastructure.runtime.geo.requests.get", side_effect=mock_get):
            result = get_country_code()
            assert result == "FR"

    def test_all_services_fail(self):
        """Test None return when all services fail."""
        clear_geo_cache()

        with patch(
            "src.infrastructure.runtime.geo.requests.get",
            side_effect=Exception("All services down"),
        ):
            result = get_country_code()
            assert result is None

    def test_invalid_country_code_length(self):
        """Test handling of invalid country code (wrong length)."""
        clear_geo_cache()
        mock_response = MagicMock()
        mock_response.text = "USA"  # 3 chars, should be 2
        mock_response.raise_for_status = MagicMock()

        # All services return invalid length
        with patch("src.infrastructure.runtime.geo.requests.get", return_value=mock_response):
            result = get_country_code()
            # Should return None since no valid 2-char code found
            assert result is None

    def test_empty_response(self):
        """Test handling of empty response."""
        clear_geo_cache()
        mock_response = MagicMock()
        mock_response.text = ""
        mock_response.raise_for_status = MagicMock()

        with patch("src.infrastructure.runtime.geo.requests.get", return_value=mock_response):
            result = get_country_code()
            assert result is None

    def test_timeout_handling(self):
        """Test that timeout is properly handled."""
        clear_geo_cache()

        import requests

        with patch(
            "src.infrastructure.runtime.geo.requests.get",
            side_effect=requests.Timeout("Connection timed out"),
        ):
            result = get_country_code()
            assert result is None

    def test_http_error_handling(self):
        """Test handling of HTTP errors."""
        clear_geo_cache()
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("404 Not Found")

        with patch("src.infrastructure.runtime.geo.requests.get", return_value=mock_response):
            result = get_country_code()
            assert result is None


class TestIsUsLocation:
    """Tests for is_us_location function."""

    def setup_method(self):
        """Clear geo cache before each test."""
        clear_geo_cache()

    def test_us_location_detected(self):
        """Test True returned for US location."""
        with patch(
            "src.infrastructure.runtime.geo.get_country_code", return_value="US"
        ):
            assert is_us_location() is True

    def test_non_us_location(self):
        """Test False returned for non-US location."""
        with patch(
            "src.infrastructure.runtime.geo.get_country_code", return_value="GB"
        ):
            assert is_us_location() is False

    def test_unknown_location(self):
        """Test False returned when location detection fails."""
        with patch(
            "src.infrastructure.runtime.geo.get_country_code", return_value=None
        ):
            assert is_us_location() is False


class TestGetBinanceApiEndpoint:
    """Tests for get_binance_api_endpoint function."""

    def setup_method(self):
        """Clear geo cache before each test."""
        clear_geo_cache()

    def test_us_endpoint(self):
        """Test binanceus endpoint for US location."""
        with patch("src.infrastructure.runtime.geo.is_us_location", return_value=True):
            result = get_binance_api_endpoint()
            assert result == "binanceus"

    def test_global_endpoint(self):
        """Test binance endpoint for non-US location."""
        with patch("src.infrastructure.runtime.geo.is_us_location", return_value=False):
            result = get_binance_api_endpoint()
            assert result == "binance"


class TestClearGeoCache:
    """Tests for clear_geo_cache function."""

    def test_clears_cache(self):
        """Test that cache is properly cleared."""
        import src.infrastructure.runtime.geo as geo_module

        geo_module._geo_cache = ("US", "test")
        clear_geo_cache()
        assert geo_module._geo_cache is None

    def test_clear_allows_redetection(self):
        """Test that clearing cache allows fresh detection."""
        import src.infrastructure.runtime.geo as geo_module

        # Set cache to US
        geo_module._geo_cache = ("US", "test")
        assert get_country_code() == "US"

        # Clear and set up mock for different country
        clear_geo_cache()
        mock_response = MagicMock()
        mock_response.text = "DE"
        mock_response.raise_for_status = MagicMock()

        with patch("src.infrastructure.runtime.geo.requests.get", return_value=mock_response):
            result = get_country_code()
            assert result == "DE"
