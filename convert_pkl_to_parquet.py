"""
One-time migration: convert all .pkl cache files to .parquet.

Run this ONCE on your local Python 3.14 machine:
    python convert_pkl_to_parquet.py

After it finishes, both the climate cache (data_storage/raw/climate/) and the
market cache (data_storage/raw/market/) will have .parquet equivalents.  The
backtest engine will then work on any Python version (3.10+) including inside
the Claude sandbox.

The original .pkl files are left untouched — you can delete them afterwards.
"""

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
CLIMATE_DIR = ROOT / "data_storage" / "raw" / "climate"
MARKET_DIR  = ROOT / "data_storage" / "raw" / "market"

def convert_dir(directory: Path, label: str) -> tuple[int, int]:
    """Convert all .pkl files in a directory to .parquet. Returns (ok, failed)."""
    pkl_files = list(directory.glob("*.pkl"))
    if not pkl_files:
        print(f"  {label}: no .pkl files found in {directory}")
        return 0, 0

    ok, failed = 0, 0
    for pkl_path in sorted(pkl_files):
        parquet_path = pkl_path.with_suffix(".parquet")
        if parquet_path.exists():
            # Already converted — skip
            ok += 1
            continue
        try:
            df = pd.read_pickle(pkl_path)
            # Ensure the index is serialisable by parquet
            if df.index.dtype.kind == "M":          # DatetimeTZDtype etc
                df = df.reset_index()
            df.to_parquet(parquet_path, index=True)
            ok += 1
            print(f"  ✓  {pkl_path.name}")
        except Exception as e:
            failed += 1
            print(f"  ✗  {pkl_path.name}  →  {e}", file=sys.stderr)

    return ok, failed


def main():
    print("=" * 60)
    print("NasaPowerCouncil — pkl → parquet cache migration")
    print("=" * 60)

    total_ok = total_fail = 0

    print(f"\n[1/2] Climate cache  ({CLIMATE_DIR})")
    ok, fail = convert_dir(CLIMATE_DIR, "climate")
    total_ok += ok; total_fail += fail

    print(f"\n[2/2] Market cache   ({MARKET_DIR})")
    ok, fail = convert_dir(MARKET_DIR, "market")
    total_ok += ok; total_fail += fail

    print("\n" + "=" * 60)
    print(f"Done.  Converted: {total_ok}   Failed: {total_fail}")
    if total_fail:
        print("Check stderr above for failed files.")
    else:
        print("All caches migrated successfully.")
        print("You can now run 'python main.py' from any Python 3.10+ environment.")
    print("=" * 60)


if __name__ == "__main__":
    main()
