"""
Geo-location detection utilities for determining the appropriate Binance API endpoint.
"""

import logging
import threading
import time

import requests

logger = logging.getLogger(__name__)

# Cache for geo-location to avoid repeated API calls
_geo_cache: tuple[str, str] | None = None
_geo_cache_lock = threading.Lock()

# Retry configuration for geo API calls
_MAX_RETRIES_PER_SERVICE = 2
_RETRY_BASE_DELAY = 0.5  # seconds


def get_country_code() -> str | None:
    """
    Get the current country code using IP geolocation.

    Returns:
        str: Two-letter country code (e.g., 'US', 'GB', 'DE') or None if detection fails.
    """
    global _geo_cache

    # Return cached result if available (double-checked locking pattern)
    with _geo_cache_lock:
        if _geo_cache is not None:
            country_code, _ = _geo_cache
            return country_code

    try:
        # Try multiple geolocation services for reliability
        services = [
            "https://ipapi.co/country/",
            "https://ipinfo.io/country",
            "https://ip-api.com/json/?fields=countryCode",
        ]

        for service_url in services:
            # Retry each service with exponential backoff
            for attempt in range(_MAX_RETRIES_PER_SERVICE):
                try:
                    response = requests.get(service_url, timeout=5)
                    response.raise_for_status()

                    if service_url.endswith("countryCode"):
                        # ip-api.com returns JSON
                        try:
                            data = response.json()
                            # Validate response is dict before accessing keys
                            if not isinstance(data, dict):
                                logger.debug(
                                    "Invalid JSON response from %s: expected dict, got %s",
                                    service_url,
                                    type(data).__name__,
                                )
                                break  # Try next service
                            country_code = data.get("countryCode", "").upper()
                        except ValueError as json_err:
                            logger.debug("Failed to parse JSON from %s: %s", service_url, json_err)
                            break  # Try next service
                    else:
                        # ipapi.co and ipinfo.io return plain text
                        country_code = response.text.strip().upper()

                    if country_code and len(country_code) == 2:
                        logger.info("Detected country code: %s using %s", country_code, service_url)
                        with _geo_cache_lock:
                            _geo_cache = (country_code, service_url)
                        return country_code
                    break  # Got response but invalid format, try next service

                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                    # Transient network errors - retry with backoff
                    if attempt < _MAX_RETRIES_PER_SERVICE - 1:
                        delay = _RETRY_BASE_DELAY * (2**attempt)
                        logger.debug(
                            "Network error from %s (attempt %d/%d): %s. Retrying in %.1fs",
                            service_url,
                            attempt + 1,
                            _MAX_RETRIES_PER_SERVICE,
                            e,
                            delay,
                        )
                        time.sleep(delay)
                        continue
                    logger.debug(
                        "Network error from %s after %d attempts: %s",
                        service_url,
                        _MAX_RETRIES_PER_SERVICE,
                        e,
                    )
                    break  # Try next service

                except Exception as e:
                    logger.debug("Failed to get country from %s: %s", service_url, e)
                    break  # Non-retryable error, try next service

        logger.warning("Failed to detect country code from all geolocation services")
        return None

    except Exception as e:
        logger.error(f"Error in geo-location detection: {e}")
        return None


def is_us_location() -> bool:
    """
    Check if the current location is in the United States.

    Returns:
        bool: True if in US, False otherwise
    """
    country_code = get_country_code()
    return country_code == "US"


def get_binance_api_endpoint() -> str:
    """
    Get the appropriate Binance API endpoint based on current location.

    Returns:
        str: Either 'binance' (global) or 'binanceus' (US) based on location
    """
    if is_us_location():
        logger.info("US location detected - using Binance US API")
        return "binanceus"
    else:
        logger.info("Non-US location detected - using global Binance API")
        return "binance"


def clear_geo_cache() -> None:
    """Clear the geo-location cache to force re-detection."""
    global _geo_cache
    with _geo_cache_lock:
        _geo_cache = None
    logger.debug("Geo-location cache cleared")
