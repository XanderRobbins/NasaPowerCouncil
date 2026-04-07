"""
Parameter sweep — tests each hyperparameter independently and reports
the best value by Sharpe ratio. Runs main.py as a subprocess with
env overrides so module-level settings reload cleanly each run.
"""
import subprocess
import os
import re
import sys

# Baseline values (current .env settings)
BASE = {
    'FORWARD_RETURN_DAYS': '21',
    'RIDGE_ALPHA': '1.0',
    'TRAIN_WINDOW_YEARS': '5',
    'SOYBEANS_VOL_REGIME_THRESHOLD': '0.20',
}

SWEEPS = {
    'FORWARD_RETURN_DAYS':           [14, 21, 28, 35, 45],
    'RIDGE_ALPHA':                   [0.1, 0.5, 1.0, 5.0, 10.0],
    'TRAIN_WINDOW_YEARS':            [3, 5, 7, 10],
    'SOYBEANS_VOL_REGIME_THRESHOLD': [0.15, 0.17, 0.20, 0.22],
}


def run_backtest(overrides: dict) -> dict:
    env = os.environ.copy()
    for k, v in BASE.items():
        env[k] = str(v)
    for k, v in overrides.items():
        env[k] = str(v)

    try:
        proc = subprocess.run(
            [sys.executable, 'main.py'],
            env=env,
            capture_output=True,
            text=True,
            timeout=360,
        )
    except subprocess.TimeoutExpired:
        print(' TIMEOUT', flush=True)
        return {'sharpe': None, 'total_return': None, 'max_dd': None}
    except Exception as e:
        print(f' ERROR: {e}', flush=True)
        return {'sharpe': None, 'total_return': None, 'max_dd': None}

    if proc.returncode != 0:
        print(f' FAILED (exit {proc.returncode})', flush=True)
        return {'sharpe': None, 'total_return': None, 'max_dd': None}

    out = proc.stdout + proc.stderr

    def extract(pattern):
        m = re.search(pattern, out)
        if m is None:
            print(f' [warn] pattern not found: {pattern}', flush=True)
            return None
        return float(m.group(1))

    return {
        'sharpe':       extract(r'Sharpe Ratio:\s+([\d.-]+)'),
        'total_return': extract(r'Total Return:\s+([\d.-]+)%'),
        'max_dd':       extract(r'Max Drawdown:\s+([\d.-]+)%'),
    }


best_params = {}

for param, values in SWEEPS.items():
    print(f"\n{'='*55}")
    print(f"Sweeping {param}")
    print(f"{'='*55}")

    rows = []
    for val in values:
        label = f"  {param}={val}"
        print(f"{label:<40}", end='', flush=True)
        metrics = run_backtest({param: val})
        s = metrics['sharpe']
        r = metrics['total_return']
        d = metrics['max_dd']
        s_str = f"{s:>5.3f}" if s is not None else "  N/A"
        r_str = f"{r:>6.2f}%" if r is not None else "   N/A"
        d_str = f"{d:>6.2f}%" if d is not None else "   N/A"
        print(f"Sharpe={s_str}  Return={r_str}  MaxDD={d_str}")
        rows.append((val, s, metrics))

    valid_rows = [(val, s, m) for val, s, m in rows if s is not None]
    if not valid_rows:
        print(f"\n  --> No valid runs for {param}, skipping")
        continue
    best_val, best_sharpe, _ = max(valid_rows, key=lambda x: x[1])
    best_params[param] = best_val
    print(f"\n  --> Best: {param}={best_val}  (Sharpe {best_sharpe:.3f})")

print(f"\n{'='*55}")
print("OPTIMAL PARAMETERS")
print(f"{'='*55}")
for k, v in best_params.items():
    changed = '  <-- changed' if str(v) != BASE[k] else ''
    print(f"  {k}={v}{changed}")
print(f"{'='*55}")
