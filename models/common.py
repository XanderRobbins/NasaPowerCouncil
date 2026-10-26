"""
Pieces shared by every model wrapper: the prediction target and feature selection.
"""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# Windowed stress features look like ``heat_stress_7d_agg``; these are the only columns
# the models are allowed to see.
FEATURE_WINDOW_TAGS = ('_7d', '_14d', '_30d')


def compute_forward_return(prices: pd.Series, horizon: int) -> pd.Series:
    """
    Target: the return ``horizon`` trading days ahead, Y_t = P_{t+h} / P_t - 1.

    The last ``horizon`` entries are NaN (their future is not known yet), so a model standing
    at time t must drop NaN labels and can only learn from rows whose outcome has happened.
    """
    return prices.pct_change(horizon).shift(-horizon)


def feature_columns(df: pd.DataFrame) -> List[str]:
    """Names of the aggregated rolling-window stress columns. Raises if there are none."""
    cols = [c for c in df.columns
            if '_agg' in c and any(tag in c for tag in FEATURE_WINDOW_TAGS)]
    if not cols:
        raise ValueError(
            "No aggregated rolling stress features found (expected columns like "
            f"'heat_stress_7d_agg'). Available columns: {df.columns.tolist()}"
        )
    return cols


def prepare_features(df: pd.DataFrame,
                     target: Optional[pd.Series],
                     max_features: int,
                     train_window: int) -> Tuple[np.ndarray, List[str]]:
    """
    Pick the model's input columns and return (X, column names).

    If there are more candidate columns than ``max_features``, keep the ``max_features``
    with the largest absolute correlation to ``target``. The correlation is computed on the
    most recent ``train_window`` rows only, pairing rows by *position* (feature row i with
    target row i). Rows whose target is still unknown (NaN) are ignored automatically.

    Missing feature values are filled with 0 ("no stress").
    """
    cols = feature_columns(df)

    if target is not None and len(cols) > max_features:
        n = min(len(df), len(target))
        lo = max(0, n - train_window)
        features_window = df[cols].iloc[lo:n].reset_index(drop=True)
        target_window = pd.Series(target.values[lo:n])

        correlations = features_window.corrwith(target_window).abs().dropna()
        if len(correlations) == 0:
            # Nothing correlates (e.g. all-constant window): fall back to the most variable columns.
            logger.warning("No usable correlations; selecting the highest-variance features instead")
            cols = df[cols].var().sort_values(ascending=False).head(max_features).index.tolist()
        else:
            cols = correlations.nlargest(max_features).index.tolist()
            logger.debug(f"Top feature correlations: {correlations.nlargest(5).to_dict()}")

    return df[cols].fillna(0).values, cols
