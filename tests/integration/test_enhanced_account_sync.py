
def test_sync_handles_mixed_legacy_and_new_orders(synchronizer, db_manager, test_session):
        """Test sync with a mix of legacy positions and new Order table records."""
        # * This test simulates the transition period where some positions
        # * have Order records and others don't
        
        # * Create a legacy position (manually set to not auto-create order for testing)
        # * We'll simulate this by creating a position directly in the DB
        legacy_position_id = db_manager.log_position(
            symbol="ETHUSDT",
            side="SHORT",
            entry_price=3000.0,
            size=0.03,
            quantity=0.01,
            strategy_name="legacy_strategy",
            entry_order_id="legacy_order_444",
            session_id=test_session
        )
        
        # * Create a new position with Order table (normal flow)
        new_position_id = db_manager.log_position(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            size=0.02,
            quantity=0.001,
            strategy_name="new_strategy",
            entry_order_id="new_order_555",
            session_id=test_session
        )
        
        # * Mock both orders on exchange
        from src.data_providers.exchange_interface import Order as ExchangeOrder
        from src.data_providers.exchange_interface import OrderSide as ExchangeOrderSide
        
        mock_orders = [
            ExchangeOrder(
                entry_order_id="legacy_order_444",
                symbol="ETHUSDT",
                side=ExchangeOrderSide.SELL,
                quantity=0.01,
                price=3000.0,
                status=ExchangeOrderStatus.FILLED
            ),
            ExchangeOrder(
                entry_order_id="new_order_555",
                symbol="BTCUSDT",
                side=ExchangeOrderSide.BUY,
                quantity=0.001,
                price=50000.0,
                status=ExchangeOrderStatus.FILLED
            )
        ]
        
        synchronizer.exchange.sync_account_data.return_value = {
            "sync_successful": True,
            "balances": [],
            "positions": [],
            "open_orders": mock_orders
        }
        
        # * Perform sync
        result = synchronizer.sync_account_data()
        assert result.success is True
        
        # * Verify both orders were processed
        order_sync_data = result.data["order_sync"]
        assert order_sync_data["synced"] is True
        assert order_sync_data["total_exchange_orders"] == 2
        assert order_sync_data["synced_orders"] == 2
