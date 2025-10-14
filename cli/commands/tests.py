import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from cli.commands.test_commands import (
    heartbeat_main,
    test_database_main,
    test_download_main,
    test_secrets_access_main,
)


def _handle_db(ns: argparse.Namespace) -> int:
    return test_database_main(ns)


def _handle_download(ns: argparse.Namespace) -> int:
    return test_download_main(ns)


def _handle_secrets(ns: argparse.Namespace) -> int:
    return test_secrets_access_main(ns)


def _collect_failures(xml_path: Path) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    try:
        tree = ET.parse(xml_path)
    except FileNotFoundError:
        print(f"JUnit XML not found: {xml_path}", file=sys.stderr)
        return failures
    except ET.ParseError as exc:
        print(f"Unable to parse {xml_path}: {exc}", file=sys.stderr)
        return failures

    root = tree.getroot()
    for testcase in root.findall(".//testcase"):
        failure = testcase.find("failure")
        if failure is None:
            continue
        classname = testcase.get("classname", "Unknown")
        name = testcase.get("name", "Unknown")
        message = failure.get("message", "No message")
        details = (failure.text or "No details").strip()
        failures.append(
            {
                "classname": classname,
                "name": name,
                "message": message,
                "details": details,
            }
        )
    return failures


def _format_failures(failures: list[dict[str, str]], test_type: str) -> str:
    if not failures:
        return f"No detailed failure information found in {test_type} XML"

    lines: list[str] = []
    for item in failures:
        lines.append(f"Test: {item['classname']}.{item['name']}")
        lines.append(f"Message: {item['message']}")
        details = item["details"]
        if len(details) > 500:
            details = details[:500] + "..."
        lines.append(f"Details: {details}")
        lines.append("---")
    return "\n".join(lines)


def _handle_parse_junit(ns: argparse.Namespace) -> int:
    xml_path = Path(ns.xml_path)
    if not xml_path.exists():
        print(f"JUnit XML file not found: {xml_path}", file=sys.stderr)
        return 1

    failures = _collect_failures(xml_path)
    output = _format_failures(failures, ns.label)
    print(output)
    return 0 if failures else 0


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("tests", help="One-off test/diagnostic scripts")
    sub = p.add_subparsers(dest="tests_cmd", required=True)

    p_db = sub.add_parser("db", help="Database test script")
    p_db.add_argument("args", nargs=argparse.REMAINDER)
    p_db.set_defaults(func=_handle_db)

    p_dl = sub.add_parser("download", help="Download test script")
    p_dl.add_argument("args", nargs=argparse.REMAINDER)
    p_dl.set_defaults(func=_handle_download)

    p_sec = sub.add_parser("secrets", help="Secrets access test script")
    p_sec.add_argument("args", nargs=argparse.REMAINDER)
    p_sec.set_defaults(func=_handle_secrets)

    def _handle_heartbeat(ns: argparse.Namespace) -> int:
        return heartbeat_main(ns)

    p_hb = sub.add_parser("heartbeat", help="Log a heartbeat SystemEvent")
    p_hb.set_defaults(func=_handle_heartbeat)

    p_junit = sub.add_parser(
        "parse-junit",
        help="Parse a JUnit XML report and print concise failure details",
    )
    p_junit.add_argument("xml_path", help="Path to the JUnit XML report")
    p_junit.add_argument(
        "--label",
        default="JUnit report",
        help="Friendly label for the report (displayed in output)",
    )
    p_junit.set_defaults(func=_handle_parse_junit)
