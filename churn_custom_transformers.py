from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import wasserstein_distance 
import pandas as pd
import numpy as np

# winsorizer class
class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for winsorization (outlier capping).

    Parameters
    ----------
    lower_q : float (default=0.01)
        Lower quantile for capping (e.g., 0.01 means 1st percentile).
    upper_q : float (default=0.99)
        Upper quantile for capping (e.g., 0.99 means 99th percentile).
    columns : list or None
        Columns to apply winsorization on.
        If None, will apply to all numeric columns during fit.
    """

    def __init__(self, lower_q=0.01, upper_q=0.99, columns=None):
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.columns = columns

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        # select numeric columns if not explicitly passed
        self.columns_ = self.columns if self.columns is not None else X.select_dtypes(include=[np.number]).columns.tolist()
        
        # store the lower/upper bounds for each column
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        for col in self.columns_:
            self.lower_bounds_[col] = X[col].quantile(self.lower_q)
            self.upper_bounds_[col] = X[col].quantile(self.upper_q)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in self.columns_:
            X[col] = np.where(X[col] < self.lower_bounds_[col], self.lower_bounds_[col], X[col])
            X[col] = np.where(X[col] > self.upper_bounds_[col], self.upper_bounds_[col], X[col])
        return X


# topncategories class
class TopNCategories(BaseEstimator, TransformerMixin):
    """
    Keep top-N most frequent categories per column; map all others to 'Other'.
    NaNs are preserved (so your imputer/encoder can handle them next).
    """
    def __init__(self, top_n=5):
        self.top_n = top_n
        self.top_categories_ = {}
        self.columns_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.columns_ = list(X.columns)
        self.top_categories_.clear()
        for col in self.columns_:
            vc = X[col].dropna().value_counts()        # frequencies (descending)
            self.top_categories_[col] = set(vc.head(self.top_n).index.tolist())
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in self.columns_:
            keep = self.top_categories_[col]
            mask = X[col].notna()
            # everything not in top-N becomes 'other'
            X.loc[mask & ~X[col].isin(keep), col] = "other"
        return X

# imputation class
class DistributionPreservingImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.fill_values_ = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            if X[col].isnull().any():
                original = X[col].dropna()
                candidates = [
                    original.min(), 
                    original.median(), 
                    original.mean(), 
                    original.max(),
                    original.quantile(0.25),
                    original.quantile(0.75)
                ]
                best_val = None
                best_dist = float('inf')
                for val in candidates:
                    temp = X[col].copy()
                    temp[X[col].isna()] = val
                    dist = wasserstein_distance(original, temp)
                    if dist < best_dist:
                        best_dist = dist
                        best_val = val
                self.fill_values_[col] = best_val
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col, fill_val in self.fill_values_.items():
            X[col] = X[col].fillna(fill_val)
        return X

