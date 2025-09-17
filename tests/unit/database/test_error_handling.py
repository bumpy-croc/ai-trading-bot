"""Error handling tests for DatabaseManager."""

import pytest

pytestmark = pytest.mark.unit


class TestErrorHandling:
    """Test error handling"""

    def test_session_rollback_on_error(self, mock_postgresql_db):
        """Test session rollback on error"""
        mock_postgresql_db._mock_session.execute.side_effect = Exception("Database error")

        with pytest.raises(Exception):  # noqa: B017
            with mock_postgresql_db.get_session() as session:
                session.execute("SELECT 1")

        mock_postgresql_db._mock_session.rollback.assert_called()

    def test_close_position_not_found(self, mock_postgresql_db):
        """Test closing non-existent position"""
        mock_query = mock_postgresql_db._mock_session.query.return_value
        mock_query.filter_by.return_value.first.return_value = None

        result = mock_postgresql_db.close_position(999)
        assert result is False
