"""Tests for atb docs commands."""

import tempfile
from pathlib import Path

from cli.commands.docs import DocValidator


class TestDocValidator:
    """Tests for the DocValidator class."""

    def test_finds_markdown_files(self):
        """Test that markdown files are found."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create test markdown files
            docs_dir = tmppath / "docs"
            docs_dir.mkdir()
            (docs_dir / "test.md").touch()

            readme = tmppath / "README.md"
            readme.touch()

            validator = DocValidator(tmppath)

            # Act
            files = validator.find_markdown_files()

            # Assert
            assert len(files) >= 2
            file_names = [f.name for f in files]
            assert "test.md" in file_names
            assert "README.md" in file_names

    def test_checks_broken_links(self):
        """Test that broken links are detected."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create markdown file with broken link
            doc = tmppath / "test.md"
            doc.write_text("[broken link](nonexistent.md)")

            validator = DocValidator(tmppath)

            # Act
            validator.check_broken_links(doc)

            # Assert
            assert len(validator.issues) > 0
            assert validator.issues[0]["type"] == "broken_link"

    def test_ignores_external_links(self):
        """Test that external links are ignored."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create markdown file with external link
            doc = tmppath / "test.md"
            doc.write_text("[external link](https://example.com)")

            validator = DocValidator(tmppath)

            # Act
            validator.check_broken_links(doc)

            # Assert
            assert len(validator.issues) == 0
