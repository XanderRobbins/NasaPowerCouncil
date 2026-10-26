"""
One-at-a-time parameter sweep: re-run ``main.py`` with one setting changed and report the
Sharpe ratio, total return and max drawdown for each value.

    python sweep.py

Each run is a separate process with environment overrides, so the module-level settings in
``config/settings.py`` reload cleanly every time.

CAUTION: every number this prints is in-sample. Picking the "best" value and quoting its
Sharpe is exactly how backtests get overfitted. With only ~10-20 seasons, differences
between neighbouring values are mostly noise. Use it to see how *sensitive* results are,
not to choose parameters.
"""
import os
import re
import subprocess
import sys

# Baseline: the defaults in config/settings.py. Each sweep changes one setting, the rest stay here.
BASE = {
    'FORWARD_RETURN_DAYS': '10',
    'RIDGE_ALPHA': '1.0',
    'TRAIN_WINDOW_YEARS': '3',
    'CORN_VOL_REGIME_THRESHOLD': '0.30',
}

# Values to try for each setting.
SWEEPS = {
    'FORWARD_RETURN_DAYS':       [10, 14, 21, 28, 35],
    'RIDGE_ALPHA':               [0.1, 0.5, 1.0, 5.0, 10.0],
    'TRAIN_WINDOW_YEARS':        [2, 3, 5, 7],
    'CORN_VOL_REGIME_THRESHOLD': [0.20, 0.25, 0.30, 0.40],
}

NO_RESULT = {'sharpe': None, 'total_return': None, 'max_dd': None}


def run_backtest(overrides: dict) -> dict:
    """Run ``main.py`` with BASE plus ``overrides`` and scrape the printed metrics."""
    env = os.environ.copy()
    env.update({k: str(v) for k, v in BASE.items()})
    env.update({k: str(v) for k, v in overrides.items()})

    try:
        proc = subprocess.run([sys.executable, 'main.py'], env=env,
                              capture_output=True, text=True, timeout=360)
    except subprocess.TimeoutExpired:
        print(' TIMEOUT', flush=True)
        return NO_RESULT
    except Exception as e:
        print(f' ERROR: {e}', flush=True)
        return NO_RESULT

    if proc.returncode != 0:
        print(f' FAILED (exit {proc.returncode})', flush=True)
        return NO_RESULT

    output = proc.stdout + proc.stderr

    def extract(pattern):
        match = re.search(pattern, output)
        if match is None:
            print(f' [warn] pattern not found: {pattern}', flush=True)
            return None
        return float(match.group(1))

    return {
        'sharpe':       extract(r'Sharpe Ratio:\s+([\d.-]+)'),
        'total_return': extract(r'Total Return:\s+([\d.-]+)%'),
        'max_dd':       extract(r'Max Drawdown:\s+([\d.-]+)%'),
    }


def main():
    best_params = {}

    for param, values in SWEEPS.items():
        print(f"\n{'=' * 55}\nSweeping {param}\n{'=' * 55}")

        rows = []
        for value in values:
            print(f"{f'  {param}={value}':<40}", end='', flush=True)
            metrics = run_backtest({param: value})
            fmt = lambda x, spec: format(x, spec) if x is not None else 'N/A'
            print(f"Sharpe={fmt(metrics['sharpe'], '>5.3f')}  "
                  f"Return={fmt(metrics['total_return'], '>6.2f')}%  "
                  f"MaxDD={fmt(metrics['max_dd'], '>6.2f')}%")
            rows.append((value, metrics['sharpe']))

        valid = [(v, s) for v, s in rows if s is not None]
        if not valid:
            print(f"\n  --> No valid runs for {param}, skipping")
            continue
        best_value, best_sharpe = max(valid, key=lambda r: r[1])
        best_params[param] = best_value
        print(f"\n  --> Highest in-sample Sharpe: {param}={best_value}  ({best_sharpe:.3f})")

    print(f"\n{'=' * 55}\nHIGHEST-SHARPE VALUES (in-sample; do not over-trust)\n{'=' * 55}")
    for param, value in best_params.items():
        marker = '  <-- differs from default' if str(value) != BASE[param] else ''
        print(f"  {param}={value}{marker}")


if __name__ == '__main__':
    main()
