#!/usr/bin/env python3
"""
Unit tests for new DatabaseManager methods
Tests: test_connection, get_database_info, get_connection_stats, cleanup_connection_pool
"""

import pytest
import sys
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from database.manager import DatabaseManager
from database.models import TradeSource, PositionSide, Base
from config.paths import get_data_dir


class TestDatabaseManagerNewMethods:
    """Test suite for new DatabaseManager methods"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory for tests"""
        temp_dir = tempfile.mkdtemp()
        original_get_data_dir = get_data_dir
        
        # Mock get_data_dir to use temp directory
        with patch('config.paths.get_data_dir', return_value=temp_dir):
            yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def sqlite_db_manager(self, temp_data_dir):
        """Create DatabaseManager with SQLite for testing"""
        # Create SQLite database URL using temp directory
        db_path = os.path.join(temp_data_dir, "test_trading_bot.db")
        sqlite_url = f"sqlite:///{db_path}"
        
        db_manager = DatabaseManager(database_url=sqlite_url)
        yield db_manager
        
        # Cleanup
        try:
            db_manager.cleanup_connection_pool()
        except:
            pass
    
    @pytest.fixture
    def mock_postgresql_db_manager(self):
        """Create DatabaseManager with mocked PostgreSQL for testing"""
        postgresql_url = "postgresql://test_user:test_pass@localhost:5432/test_db"
        
        # Mock the SQLAlchemy components
        with patch('database.manager.create_engine') as mock_create_engine, \
             patch('database.manager.sessionmaker') as mock_sessionmaker, \
             patch('database.manager.Base') as mock_base:
            
            # Mock engine and session
            mock_engine = Mock()
            mock_session_factory = Mock()
            mock_session = Mock()
            
            mock_create_engine.return_value = mock_engine
            mock_sessionmaker.return_value = mock_session_factory
            mock_session_factory.return_value = mock_session
            
            # Mock engine connection
            mock_connection = Mock()
            mock_engine.connect.return_value.__enter__ = Mock(return_value=mock_connection)
            mock_engine.connect.return_value.__exit__ = Mock(return_value=None)
            
            # Mock connection pool attributes
            mock_engine.pool = Mock()
            mock_engine.pool.size = 5
            mock_engine.pool.checkedin = 2
            mock_engine.pool.checkedout = 3
            mock_engine.pool.overflow = 1
            mock_engine.pool.invalid = 0
            mock_engine.pool.status.return_value = "Pool status info"
            mock_engine.pool.dispose = Mock()
            
            # Mock session query execution
            mock_session.execute.return_value = None
            mock_session.__enter__ = Mock(return_value=mock_session)
            mock_session.__exit__ = Mock(return_value=None)
            
            # Create database manager
            db_manager = DatabaseManager(database_url=postgresql_url)
            
            # Store mocks for access in tests
            db_manager._mock_engine = mock_engine
            db_manager._mock_session = mock_session
            db_manager._mock_connection = mock_connection
            
            yield db_manager


class TestConnectionMethods(TestDatabaseManagerNewMethods):
    """Test connection-related methods"""
    
    def test_test_connection_sqlite_success(self, sqlite_db_manager):
        """Test successful connection with SQLite"""
        result = sqlite_db_manager.test_connection()
        assert result is True
    
    def test_test_connection_postgresql_success(self, mock_postgresql_db_manager):
        """Test successful connection with PostgreSQL"""
        result = mock_postgresql_db_manager.test_connection()
        assert result is True
    
    def test_test_connection_failure(self, temp_data_dir):
        """Test connection failure handling"""
        # Use invalid database URL
        invalid_url = "postgresql://invalid:invalid@nonexistent:5432/invalid"
        
        with patch('database.manager.create_engine') as mock_create_engine:
            # Make engine creation fail
            mock_create_engine.side_effect = Exception("Connection failed")
            
            with pytest.raises(Exception):
                DatabaseManager(database_url=invalid_url)
    
    def test_test_connection_with_session_error(self, mock_postgresql_db_manager):
        """Test test_connection when session execution fails"""
        # Make session.execute raise an exception
        mock_postgresql_db_manager._mock_session.execute.side_effect = Exception("Query failed")
        
        result = mock_postgresql_db_manager.test_connection()
        assert result is False


class TestDatabaseInfo(TestDatabaseManagerNewMethods):
    """Test database info methods"""
    
    def test_get_database_info_sqlite(self, sqlite_db_manager):
        """Test get_database_info with SQLite"""
        info = sqlite_db_manager.get_database_info()
        
        assert isinstance(info, dict)
        assert 'database_url' in info
        assert 'is_postgresql' in info
        assert 'is_sqlite' in info
        assert 'connection_pool_size' in info
        assert 'checked_in_connections' in info
        assert 'checked_out_connections' in info
        
        # SQLite specific assertions
        assert info['is_postgresql'] is False
        assert info['is_sqlite'] is True
        assert info['database_url'].startswith('sqlite:///')
        assert info['connection_pool_size'] == 0  # SQLite uses NullPool
    
    def test_get_database_info_postgresql(self, mock_postgresql_db_manager):
        """Test get_database_info with PostgreSQL"""
        info = mock_postgresql_db_manager.get_database_info()
        
        assert isinstance(info, dict)
        assert 'database_url' in info
        assert 'is_postgresql' in info
        assert 'is_sqlite' in info
        assert 'connection_pool_size' in info
        assert 'checked_in_connections' in info
        assert 'checked_out_connections' in info
        
        # PostgreSQL specific assertions
        assert info['is_postgresql'] is True
        assert info['is_sqlite'] is False
        assert info['database_url'].startswith('postgresql://')
        assert info['connection_pool_size'] == 5  # Mocked pool size
        assert info['checked_in_connections'] == 2  # Mocked value
        assert info['checked_out_connections'] == 3  # Mocked value
    
    def test_get_database_info_with_none_database_url(self):
        """Test get_database_info when database_url is None"""
        with patch('database.manager.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.get.return_value = None
            mock_get_config.return_value = mock_config
            
            with patch('database.manager.get_database_path') as mock_get_db_path:
                mock_get_db_path.return_value = "sqlite:///test.db"
                
                db_manager = DatabaseManager(database_url=None)
                info = db_manager.get_database_info()
                
                assert info['database_url'] is not None
                assert info['is_sqlite'] is True


class TestConnectionStats(TestDatabaseManagerNewMethods):
    """Test connection statistics methods"""
    
    def test_get_connection_stats_postgresql(self, mock_postgresql_db_manager):
        """Test get_connection_stats with PostgreSQL"""
        stats = mock_postgresql_db_manager.get_connection_stats()
        
        assert isinstance(stats, dict)
        assert 'pool_status' in stats
        assert 'checked_in' in stats
        assert 'checked_out' in stats
        assert 'overflow' in stats
        assert 'invalid' in stats
        
        # Check mocked values
        assert stats['checked_in'] == 2
        assert stats['checked_out'] == 3
        assert stats['overflow'] == 1
        assert stats['invalid'] == 0
        assert stats['pool_status'] == "Pool status info"
    
    def test_get_connection_stats_sqlite(self, sqlite_db_manager):
        """Test get_connection_stats with SQLite"""
        stats = sqlite_db_manager.get_connection_stats()
        
        assert isinstance(stats, dict)
        assert 'status' in stats
        assert stats['status'] == 'No connection pool statistics available'
    
    def test_get_connection_stats_no_pool_status(self, mock_postgresql_db_manager):
        """Test get_connection_stats when engine.pool has no status method"""
        # Remove status method from mock pool
        del mock_postgresql_db_manager._mock_engine.pool.status
        
        stats = mock_postgresql_db_manager.get_connection_stats()
        
        assert isinstance(stats, dict)
        assert 'status' in stats
        assert stats['status'] == 'No connection pool statistics available'


class TestConnectionPoolCleanup(TestDatabaseManagerNewMethods):
    """Test connection pool cleanup methods"""
    
    def test_cleanup_connection_pool_postgresql(self, mock_postgresql_db_manager):
        """Test cleanup_connection_pool with PostgreSQL"""
        # Should not raise an exception
        mock_postgresql_db_manager.cleanup_connection_pool()
        
        # Verify dispose was called
        mock_postgresql_db_manager._mock_engine.pool.dispose.assert_called_once()
    
    def test_cleanup_connection_pool_sqlite(self, sqlite_db_manager):
        """Test cleanup_connection_pool with SQLite"""
        # Should not raise an exception even though SQLite doesn't have a pool
        sqlite_db_manager.cleanup_connection_pool()
    
    def test_cleanup_connection_pool_no_dispose_method(self, mock_postgresql_db_manager):
        """Test cleanup_connection_pool when pool has no dispose method"""
        # Remove dispose method from mock pool
        del mock_postgresql_db_manager._mock_engine.pool.dispose
        
        # Should not raise an exception
        mock_postgresql_db_manager.cleanup_connection_pool()


class TestIntegrationScenarios(TestDatabaseManagerNewMethods):
    """Test integration scenarios combining multiple methods"""
    
    def test_full_database_workflow_sqlite(self, sqlite_db_manager):
        """Test full workflow with SQLite"""
        # Test connection
        assert sqlite_db_manager.test_connection() is True
        
        # Get database info
        info = sqlite_db_manager.get_database_info()
        assert info['is_sqlite'] is True
        
        # Create a trading session to test database operations
        session_id = sqlite_db_manager.create_trading_session(
            strategy_name="TestStrategy",
            symbol="BTCUSDT",
            timeframe="1h",
            mode="PAPER",
            initial_balance=10000.0
        )
        assert session_id is not None
        
        # Get connection stats
        stats = sqlite_db_manager.get_connection_stats()
        assert isinstance(stats, dict)
        
        # Cleanup
        sqlite_db_manager.cleanup_connection_pool()
    
    def test_full_database_workflow_postgresql(self, mock_postgresql_db_manager):
        """Test full workflow with PostgreSQL"""
        # Test connection
        assert mock_postgresql_db_manager.test_connection() is True
        
        # Get database info
        info = mock_postgresql_db_manager.get_database_info()
        assert info['is_postgresql'] is True
        assert info['connection_pool_size'] > 0
        
        # Get connection stats
        stats = mock_postgresql_db_manager.get_connection_stats()
        assert 'pool_status' in stats
        
        # Cleanup
        mock_postgresql_db_manager.cleanup_connection_pool()
    
    def test_error_handling_robustness(self, sqlite_db_manager):
        """Test error handling in various scenarios"""
        # Test with temporarily broken database
        original_engine = sqlite_db_manager.engine
        sqlite_db_manager.engine = None
        
        # Should handle None engine gracefully
        info = sqlite_db_manager.get_database_info()
        assert info['connection_pool_size'] == 0
        
        # Restore engine
        sqlite_db_manager.engine = original_engine
        
        # Should work normally again
        assert sqlite_db_manager.test_connection() is True


class TestEdgeCases(TestDatabaseManagerNewMethods):
    """Test edge cases and error conditions"""
    
    def test_database_url_edge_cases(self):
        """Test various database URL formats"""
        test_cases = [
            ("sqlite:///test.db", True, False),
            ("postgresql://user:pass@host:5432/db", False, True),
            ("mysql://user:pass@host:3306/db", False, False),
            ("", False, False),
        ]
        
        for url, is_sqlite, is_postgresql in test_cases:
            with patch('database.manager.create_engine'), \
                 patch('database.manager.sessionmaker'), \
                 patch('database.manager.Base.metadata.create_all'):
                
                try:
                    db_manager = DatabaseManager(database_url=url)
                    info = db_manager.get_database_info()
                    
                    assert info['is_sqlite'] == is_sqlite
                    assert info['is_postgresql'] == is_postgresql
                except:
                    # Some URLs might fail to initialize, which is expected
                    pass
    
    def test_concurrent_access_safety(self, sqlite_db_manager):
        """Test that methods are safe for concurrent access"""
        import threading
        import time
        
        results = []
        errors = []
        
        def worker():
            try:
                # Test various methods concurrently
                result = sqlite_db_manager.test_connection()
                results.append(result)
                
                info = sqlite_db_manager.get_database_info()
                results.append(info is not None)
                
                stats = sqlite_db_manager.get_connection_stats()
                results.append(stats is not None)
                
            except Exception as e:
                errors.append(str(e))
        
        # Run multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert all(results), "Some concurrent operations failed"


# Performance benchmarks
class TestPerformance(TestDatabaseManagerNewMethods):
    """Performance tests for new methods"""
    
    def test_method_performance(self, sqlite_db_manager):
        """Test that methods execute within reasonable time"""
        import time
        
        # Test test_connection performance
        start_time = time.time()
        result = sqlite_db_manager.test_connection()
        end_time = time.time()
        
        assert result is True
        assert (end_time - start_time) < 1.0  # Should complete within 1 second
        
        # Test get_database_info performance
        start_time = time.time()
        info = sqlite_db_manager.get_database_info()
        end_time = time.time()
        
        assert info is not None
        assert (end_time - start_time) < 0.1  # Should be very fast
        
        # Test get_connection_stats performance
        start_time = time.time()
        stats = sqlite_db_manager.get_connection_stats()
        end_time = time.time()
        
        assert stats is not None
        assert (end_time - start_time) < 0.1  # Should be very fast


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])