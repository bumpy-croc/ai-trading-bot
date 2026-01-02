"""Tests for atb railway commands."""

from unittest.mock import Mock, patch

from cli.commands.railway import _authenticate_railway


class TestAuthenticateRailway:
    """Tests for the _authenticate_railway function."""

    def test_authenticates_successfully(self):
        """Test that Railway authentication succeeds."""
        # Arrange & Act
        with (
            patch("cli.commands.railway.subprocess.run") as mock_run,
            patch.dict("os.environ", {"RAILWAY_PROJECT_ID": "test-project-id"}),
        ):

            # Mock successful whoami
            mock_whoami = Mock(returncode=0, stdout="user@example.com\n")

            # Mock successful list
            mock_list = Mock(
                returncode=0, stdout='[{"id": "test-project-id", "name": "test-project"}]'
            )

            mock_run.side_effect = [mock_whoami, mock_list]

            result = _authenticate_railway()

            # Assert
            assert result == "test-project-id"

    def test_returns_none_when_not_authenticated(self):
        """Test that None is returned when not authenticated."""
        # Arrange & Act
        with patch("cli.commands.railway.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=1, stderr="Not authenticated")

            result = _authenticate_railway()

            # Assert
            assert result is None

    def test_returns_none_when_project_id_missing(self):
        """Test that None is returned when project ID is missing."""
        # Arrange & Act
        with (
            patch("cli.commands.railway.subprocess.run") as mock_run,
            patch.dict("os.environ", {}, clear=True),
        ):

            mock_run.return_value = Mock(returncode=0)

            result = _authenticate_railway()

            # Assert
            assert result is None
