#!/usr/bin/env python3
"""
Cache management utility for the trading bot.
"""

import argparse
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.paths import get_cache_dir
from data_providers.binance_provider import BinanceProvider
from data_providers.cached_data_provider import CachedDataProvider


def format_file_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f} {size_names[i]}"


def show_cache_info(cache_dir=None):
    """Show cache information"""
    if cache_dir is None:
        cache_dir = str(get_cache_dir())
    provider = CachedDataProvider(BinanceProvider(), cache_dir=cache_dir)
    info = provider.get_cache_info()

    print("Cache Information:")
    print("=" * 40)
    print(f"Cache Directory: {cache_dir}")
    print(f"Total Files: {info['total_files']}")
    print(f"Total Size: {format_file_size(info['total_size_mb'] * 1024 * 1024)}")

    if info["oldest_file"]:
        print(f"Oldest File: {info['oldest_file']}")
    if info["newest_file"]:
        print(f"Newest File: {info['newest_file']}")

    return info


def list_cache_files(cache_dir=None, detailed=False):
    """List all cache files"""
    if cache_dir is None:
        cache_dir = str(get_cache_dir())

    if not os.path.exists(cache_dir):
        print(f"Cache directory {cache_dir} does not exist.")
        return

    files = [f for f in os.listdir(cache_dir) if f.endswith(".pkl")]

    if not files:
        print("No cache files found.")
        return

    print(f"\nCache Files ({len(files)} total):")
    print("=" * 60)

    file_info = []
    for filename in files:
        file_path = os.path.join(cache_dir, filename)
        size = os.path.getsize(file_path)
        mtime = datetime.fromtimestamp(os.path.getmtime(file_path))

        file_info.append({"name": filename, "size": size, "modified": mtime, "path": file_path})

    # Sort by modification time (newest first)
    file_info.sort(key=lambda x: x["modified"], reverse=True)

    for info in file_info:
        if detailed:
            try:
                # Try to load and inspect the cached data
                with open(info["path"], "rb") as f:
                    data = pickle.load(f)

                data_info = ""
                if hasattr(data, "shape"):
                    data_info = f" - {data.shape[0]} rows"
                if hasattr(data, "index") and len(data.index) > 0:
                    start_date = data.index.min().strftime("%Y-%m-%d")
                    end_date = data.index.max().strftime("%Y-%m-%d")
                    data_info += f" ({start_date} to {end_date})"

                print(
                    f"{info['name'][:20]:<20} {format_file_size(info['size']):<8} {info['modified'].strftime('%Y-%m-%d %H:%M')}{data_info}"
                )
            except Exception as e:
                print(
                    f"{info['name'][:20]:<20} {format_file_size(info['size']):<8} {info['modified'].strftime('%Y-%m-%d %H:%M')} - Error reading: {e}"
                )
        else:
            print(
                f"{info['name'][:20]:<20} {format_file_size(info['size']):<8} {info['modified'].strftime('%Y-%m-%d %H:%M')}"
            )


def clear_cache(cache_dir=None, confirm=True):
    """Clear all cache files"""
    if cache_dir is None:
        cache_dir = str(get_cache_dir())

    if not os.path.exists(cache_dir):
        print(f"Cache directory {cache_dir} does not exist.")
        return

    files = [f for f in os.listdir(cache_dir) if f.endswith(".pkl")]

    if not files:
        print("No cache files to clear.")
        return

    if confirm:
        response = input(f"Are you sure you want to delete {len(files)} cache files? (y/N): ")
        if response.lower() != "y":
            print("Cache clear cancelled.")
            return

    deleted_count = 0
    total_size = 0

    for filename in files:
        file_path = os.path.join(cache_dir, filename)
        try:
            size = os.path.getsize(file_path)
            os.remove(file_path)
            deleted_count += 1
            total_size += size
        except Exception as e:
            print(f"Error deleting {filename}: {e}")

    print(f"Deleted {deleted_count} cache files, freed {format_file_size(total_size)}")


def clear_old_cache(cache_dir=None, hours=24):
    """Clear cache files older than specified hours"""
    if cache_dir is None:
        cache_dir = str(get_cache_dir())

    if not os.path.exists(cache_dir):
        print(f"Cache directory {cache_dir} does not exist.")
        return

    files = [f for f in os.listdir(cache_dir) if f.endswith(".pkl")]
    current_time = datetime.now()

    deleted_count = 0
    total_size = 0

    for filename in files:
        file_path = os.path.join(cache_dir, filename)
        try:
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            age_hours = (current_time - file_time).total_seconds() / 3600

            if age_hours > hours:
                size = os.path.getsize(file_path)
                os.remove(file_path)
                deleted_count += 1
                total_size += size
                print(f"Deleted {filename} (age: {age_hours:.1f} hours)")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if deleted_count > 0:
        print(f"Deleted {deleted_count} old cache files, freed {format_file_size(total_size)}")
    else:
        print(f"No cache files older than {hours} hours found.")


def main():
    parser = argparse.ArgumentParser(description="Manage data cache")
    parser.add_argument("--cache-dir", default=None, help="Cache directory path")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Info command
    subparsers.add_parser("info", help="Show cache information")

    # List command
    list_parser = subparsers.add_parser("list", help="List cache files")
    list_parser.add_argument("--detailed", action="store_true", help="Show detailed information")

    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear all cache files")
    clear_parser.add_argument("--force", action="store_true", help="Skip confirmation")

    # Clear old command
    clear_old_parser = subparsers.add_parser("clear-old", help="Clear old cache files")
    clear_old_parser.add_argument(
        "--hours", type=int, default=24, help="Clear files older than N hours"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "info":
        show_cache_info(args.cache_dir)

    elif args.command == "list":
        list_cache_files(args.cache_dir, args.detailed)

    elif args.command == "clear":
        clear_cache(args.cache_dir, not args.force)

    elif args.command == "clear-old":
        clear_old_cache(args.cache_dir, args.hours)


if __name__ == "__main__":
    main()
