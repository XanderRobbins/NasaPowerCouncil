"""
Direction classifier: logistic regression predicting whether the forward return is positive.

It is used only as a veto. When it confidently disagrees with the regression's direction,
the engine skips the trade (see ``VETO_CONFIDENCE`` in ``config/settings.py``). It is fitted
by the engine on the same scaled features as the regression.
"""
from sklearn.linear_model import LogisticRegression


class DirectionalClassifier:
    """Thin holder for the sklearn logistic regression (``self.model``)."""

    def __init__(self, C: float = 1.0):
        self.model = LogisticRegression(C=C, max_iter=1000, random_state=42)
