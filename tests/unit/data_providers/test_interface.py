import pytest

from data_providers.data_provider import DataProvider

pytestmark = pytest.mark.unit


class TestDataProviderInterface:
    def test_data_provider_is_abstract(self):
        with pytest.raises(TypeError):
            DataProvider()

    def test_data_provider_interface_methods(self):
        required_methods = [
            "get_historical_data",
            "get_live_data",
            "update_live_data",
            "get_current_price",
        ]
        for method in required_methods:
            assert hasattr(DataProvider, method)
