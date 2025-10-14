from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

# Ensure project root and src are importable when the command is packaged
from src.utils.project_paths import get_project_root

PROJECT_ROOT = get_project_root()


class DocValidator:
    def __init__(self, root_path: Path):
        self.root = root_path
        self.issues: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []

    def find_markdown_files(self) -> list[Path]:
        files = []
        files.extend(self.root.glob("docs/**/*.md"))
        files.extend(self.root.glob("src/**/README.md"))
        readme = self.root / "README.md"
        if readme.exists():
            files.append(readme)
        return sorted({f.resolve() for f in files})

    def check_broken_links(self, file_path: Path) -> None:
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            self.warnings.append(
                {
                    "type": "read_error",
                    "file": str(file_path.relative_to(self.root)),
                    "message": str(exc),
                }
            )
            return

        link_pattern = re.compile(r"\[([^\]]+)\]\(([^\)]+)\)")
        for match in link_pattern.finditer(content):
            link_url = match.group(2)
            if link_url.startswith(("http://", "https://", "#", "mailto:")):
                continue
            file_url = link_url.split("#")[0]
            if not file_url:
                continue
            target = (file_path.parent / file_url).resolve()
            if not target.exists():
                try:
                    rel = target.relative_to(self.root)
                    target_display = str(rel)
                except ValueError:
                    target_display = str(target)
                self.issues.append(
                    {
                        "type": "broken_link",
                        "severity": "error",
                        "file": str(file_path.relative_to(self.root)),
                        "link": link_url,
                        "target": target_display,
                        "line": content[: match.start()].count("\n") + 1,
                    }
                )

    def check_outdated_commands(self, file_path: Path) -> None:
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            return
        lines = content.split("\n")
        for idx, line in enumerate(lines, 1):
            if "docker-compose" in line.lower() and "`docker-compose`" not in line:
                self.issues.append(
                    {
                        "type": "outdated_command",
                        "severity": "warning",
                        "file": str(file_path.relative_to(self.root)),
                        "line": idx,
                        "issue": "Use 'docker compose' instead of 'docker-compose'",
                        "content": line.strip()[:80],
                    }
                )
        for idx, line in enumerate(lines, 1):
            if "sqlite" in line.lower() and "removed" not in line.lower():
                if "fallback" in line.lower() or "support" in line.lower():
                    self.warnings.append(
                        {
                            "type": "sqlite_reference",
                            "severity": "warning",
                            "file": str(file_path.relative_to(self.root)),
                            "line": idx,
                            "issue": "SQLite reference detected (project is PostgreSQL-only)",
                            "content": line.strip()[:80],
                        }
                    )

    def check_module_readmes(self) -> None:
        src_dir = self.root / "src"
        if not src_dir.exists():
            return
        for item in src_dir.iterdir():
            if item.is_dir() and not item.name.startswith((".", "_")):
                readme = item / "README.md"
                if not readme.exists():
                    self.issues.append(
                        {
                            "type": "missing_readme",
                            "severity": "warning",
                            "file": str(item.relative_to(self.root)),
                            "message": "Module missing README.md",
                        }
                    )

    def check_code_examples(self, file_path: Path) -> None:
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            return
        code_pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
        for match in code_pattern.finditer(content):
            code = match.group(1)
            if len(code.strip()) < 20:
                continue
            if code.strip().startswith(("from src.", "import src")):
                if code.count("(") != code.count(")"):
                    self.warnings.append(
                        {
                            "type": "unbalanced_parens",
                            "severity": "info",
                            "file": str(file_path.relative_to(self.root)),
                            "line": content[: match.start()].count("\n") + 1,
                            "message": "Unbalanced parentheses in code example",
                        }
                    )

    def validate_cli_commands(self, file_path: Path) -> None:
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            return
        known_commands = {
            "backtest",
            "live",
            "live-health",
            "dashboards",
            "data",
            "db",
            "optimizer",
            "tests",
            "train",
            "dev",
            "models",
            "migration",
            "docs",
            "strategies",
            "regime",
        }
        atb_pattern = re.compile(r"atb\s+([a-z-]+)")
        for match in atb_pattern.finditer(content):
            cmd = match.group(1)
            if cmd not in known_commands:
                # Could be a subcommand name; skip for now
                continue

    def run_validation(self) -> tuple[int, int]:
        print("🔍 Starting documentation validation...\n")
        md_files = self.find_markdown_files()
        print(f"Found {len(md_files)} markdown files to validate")
        for md_file in md_files:
            self.check_broken_links(md_file)
            self.check_outdated_commands(md_file)
            self.check_code_examples(md_file)
            self.validate_cli_commands(md_file)
        self.check_module_readmes()
        return len(self.issues), len(self.warnings)

    def _print_issue(self, issue: dict[str, Any]) -> None:
        issue_type = issue["type"].replace("_", " ").title()
        file_path = issue.get("file", "unknown")
        line = issue.get("line", "")
        location = f"{file_path}:{line}" if line else file_path
        print(f"  [{issue_type}] {location}")
        if "message" in issue:
            print(f"    {issue['message']}")
        if "link" in issue:
            print(f"    Broken link: {issue['link']}")
        if "target" in issue:
            print(f"    Target not found: {issue['target']}")
        if "issue" in issue:
            print(f"    Issue: {issue['issue']}")
        if "content" in issue:
            print(f"    Content: {issue['content']}")
        print()

    def print_report(self) -> int:
        if not self.issues and not self.warnings:
            print("\n✅ All documentation validation checks passed!")
            return 0
        errors = [i for i in self.issues if i.get("severity") == "error"]
        warnings = [i for i in self.issues if i.get("severity") == "warning"] + self.warnings
        if errors:
            print(f"\n❌ Found {len(errors)} error(s):\n")
            for issue in errors:
                self._print_issue(issue)
        if warnings:
            print(f"\n⚠️  Found {len(warnings)} warning(s):\n")
            for issue in warnings[:20]:
                self._print_issue(issue)
            if len(warnings) > 20:
                print(f"   ... and {len(warnings) - 20} more warnings")
        return 1 if errors else 0


def _handle_validate(ns: argparse.Namespace) -> int:
    validator = DocValidator(PROJECT_ROOT)
    error_count, warning_count = validator.run_validation()
    print(f"\n📊 Summary: {error_count} error(s), {warning_count} warning(s)")
    return validator.print_report()


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("docs", help="Documentation utilities")
    sub = parser.add_subparsers(dest="docs_cmd", required=True)
    p_validate = sub.add_parser("validate", help="Validate markdown documentation")
    p_validate.set_defaults(func=_handle_validate)
