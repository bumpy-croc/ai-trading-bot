"""
Geo-location detection utilities for determining the appropriate Binance API endpoint.
"""

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Cache for geo-location to avoid repeated API calls
_geo_cache: Optional[tuple[str, str]] = None


def get_country_code() -> Optional[str]:
    """
    Get the current country code using IP geolocation.

    Returns:
        str: Two-letter country code (e.g., 'US', 'GB', 'DE') or None if detection fails
    """
    global _geo_cache

    # Return cached result if available
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
            try:
                response = requests.get(service_url, timeout=5)
                response.raise_for_status()

                if service_url.endswith("countryCode"):
                    # ip-api.com returns JSON
                    data = response.json()
                    country_code = data.get("countryCode", "").upper()
                else:
                    # ipapi.co and ipinfo.io return plain text
                    country_code = response.text.strip().upper()

                if country_code and len(country_code) == 2:
                    logger.info(f"Detected country code: {country_code} using {service_url}")
                    _geo_cache = (country_code, service_url)
                    return country_code

            except Exception as e:
                logger.debug(f"Failed to get country from {service_url}: {e}")
                continue

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


def clear_geo_cache():
    """Clear the geo-location cache to force re-detection."""
    global _geo_cache
    _geo_cache = None
    logger.debug("Geo-location cache cleared")
