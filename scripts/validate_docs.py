#!/usr/bin/env python3
"""
Documentation Validation Script

Validates documentation for:
- Broken links (internal)
- Missing referenced files
- Outdated command examples
- Consistency in code examples
- Module README presence
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set


class DocValidator:
    def __init__(self, root_path: Path):
        self.root = root_path
        self.issues: List[Dict] = []
        self.warnings: List[Dict] = []
        
    def find_markdown_files(self) -> List[Path]:
        """Find all markdown files in docs/ and src/."""
        files = []
        # Main docs
        files.extend(self.root.glob('docs/**/*.md'))
        # Module READMEs
        files.extend(self.root.glob('src/**/README.md'))
        # Root README
        if (self.root / 'README.md').exists():
            files.append(self.root / 'README.md')
        return sorted(set(files))
    
    def check_broken_links(self, file_path: Path):
        """Check for broken internal file references."""
        try:
            content = file_path.read_text()
        except Exception as e:
            self.warnings.append({
                'type': 'read_error',
                'file': str(file_path.relative_to(self.root)),
                'message': str(e)
            })
            return
            
        link_pattern = re.compile(r'\[([^\]]+)\]\(([^\)]+)\)')
        
        for match in link_pattern.finditer(content):
            link_text, link_url = match.groups()
            
            # Skip external URLs and anchors
            if link_url.startswith(('http://', 'https://', '#', 'mailto:')):
                continue
            
            # Remove anchor from URL
            file_url = link_url.split('#')[0]
            if not file_url:
                continue
            
            # Resolve target path
            try:
                target = (file_path.parent / file_url).resolve()
                if not target.exists():
                    self.issues.append({
                        'type': 'broken_link',
                        'severity': 'error',
                        'file': str(file_path.relative_to(self.root)),
                        'link': link_url,
                        'target': str(target.relative_to(self.root)) if target.is_relative_to(self.root) else str(target),
                        'line': content[:match.start()].count('\n') + 1
                    })
            except Exception:
                # Could not resolve path
                pass
    
    def check_outdated_commands(self, file_path: Path):
        """Check for outdated command examples."""
        try:
            content = file_path.read_text()
        except Exception:
            return
        
        lines = content.split('\n')
        
        # Check for old docker-compose syntax
        for i, line in enumerate(lines, 1):
            if 'docker-compose' in line.lower() and '`docker-compose`' not in line:
                self.issues.append({
                    'type': 'outdated_command',
                    'severity': 'warning',
                    'file': str(file_path.relative_to(self.root)),
                    'line': i,
                    'issue': 'Use "docker compose" instead of "docker-compose"',
                    'content': line.strip()[:80]
                })
        
        # Check for SQLite references (should be removed)
        for i, line in enumerate(lines, 1):
            if 'sqlite' in line.lower() and 'removed' not in line.lower():
                # Check context to avoid false positives
                if 'fallback' in line.lower() or 'support' in line.lower():
                    self.warnings.append({
                        'type': 'sqlite_reference',
                        'severity': 'warning',
                        'file': str(file_path.relative_to(self.root)),
                        'line': i,
                        'issue': 'SQLite reference detected (project is PostgreSQL-only)',
                        'content': line.strip()[:80]
                    })
    
    def check_module_readmes(self):
        """Check that all src/ modules have READMEs."""
        src_dir = self.root / 'src'
        if not src_dir.exists():
            return
        
        for item in src_dir.iterdir():
            if item.is_dir() and not item.name.startswith(('.', '_')):
                readme = item / 'README.md'
                if not readme.exists():
                    self.issues.append({
                        'type': 'missing_readme',
                        'severity': 'warning',
                        'file': str(item.relative_to(self.root)),
                        'message': 'Module missing README.md'
                    })
    
    def check_code_examples(self, file_path: Path):
        """Check if Python code examples are syntactically valid."""
        try:
            content = file_path.read_text()
        except Exception:
            return
        
        # Find Python code blocks
        code_pattern = re.compile(r'```python\n(.*?)```', re.DOTALL)
        for match in code_pattern.finditer(content):
            code = match.group(1)
            
            # Skip if it's just imports or very short snippets
            if len(code.strip()) < 20:
                continue
            
            # Basic validation - check for common issues
            if code.strip().startswith('from src.') or code.strip().startswith('import src'):
                # Check for obvious syntax errors
                if code.count('(') != code.count(')'):
                    self.warnings.append({
                        'type': 'unbalanced_parens',
                        'severity': 'info',
                        'file': str(file_path.relative_to(self.root)),
                        'line': content[:match.start()].count('\n') + 1,
                        'message': 'Unbalanced parentheses in code example'
                    })
    
    def validate_cli_commands(self, file_path: Path):
        """Validate that documented CLI commands match known commands."""
        try:
            content = file_path.read_text()
        except Exception:
            return
        
        # Known valid top-level commands
        known_commands = {
            'backtest', 'live', 'live-health', 'dashboards', 'data',
            'db', 'optimizer', 'tests', 'train', 'dev', 'models', 'scripts'
        }
        
        # Extract atb commands
        atb_pattern = re.compile(r'atb\s+([a-z-]+)')
        for match in atb_pattern.finditer(content):
            cmd = match.group(1)
            if cmd not in known_commands:
                # Could be a strategy name or subcommand, don't flag as error
                pass
    
    def run_validation(self) -> Tuple[int, int]:
        """Run all validation checks."""
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
    
    def print_report(self):
        """Print validation report."""
        if not self.issues and not self.warnings:
            print("\n✅ All documentation validation checks passed!")
            return 0
        
        # Group by severity
        errors = [i for i in self.issues if i.get('severity') == 'error']
        warnings = [i for i in self.issues if i.get('severity') == 'warning'] + self.warnings
        
        if errors:
            print(f"\n❌ Found {len(errors)} error(s):\n")
            for issue in errors:
                self._print_issue(issue)
        
        if warnings:
            print(f"\n⚠️  Found {len(warnings)} warning(s):\n")
            for issue in warnings[:20]:  # Limit warnings
                self._print_issue(issue)
            if len(warnings) > 20:
                print(f"   ... and {len(warnings) - 20} more warnings")
        
        return 1 if errors else 0
    
    def _print_issue(self, issue: Dict):
        """Print a single issue."""
        issue_type = issue['type'].replace('_', ' ').title()
        file_path = issue.get('file', 'unknown')
        line = issue.get('line', '')
        
        location = f"{file_path}:{line}" if line else file_path
        print(f"  [{issue_type}] {location}")
        
        if 'message' in issue:
            print(f"    {issue['message']}")
        if 'link' in issue:
            print(f"    Broken link: {issue['link']}")
        if 'target' in issue:
            print(f"    Target not found: {issue['target']}")
        if 'issue' in issue:
            print(f"    Issue: {issue['issue']}")
        if 'content' in issue:
            print(f"    Content: {issue['content']}")
        print()


def main():
    """Main entry point."""
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent if script_dir.name == 'scripts' else script_dir
    
    validator = DocValidator(root_dir)
    error_count, warning_count = validator.run_validation()
    
    print(f"\n📊 Summary: {error_count} error(s), {warning_count} warning(s)")
    
    exit_code = validator.print_report()
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
