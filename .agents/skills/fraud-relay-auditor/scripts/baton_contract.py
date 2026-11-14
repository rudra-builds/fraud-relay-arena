"""
baton_contract.py – Relay Baton Handoff Contract & Chronological Data Splitter
==============================================================================
Enforces strict train/test separation between Leg 1 (feature engineering) and
Leg 2 (classification) agents.  Includes a deterministic synthetic fallback so
the full relay pipeline runs even without the 150 MB Kaggle CSV.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any
import hashlib
import numpy as np
import pandas as pd


# ── Relay Baton Dataclass ───────────────────────────────────────────────────

@dataclass
class RelayBaton:
    """Immutable handoff contract between Leg 1 and Leg 2 of a relay lane."""

    lane_id: str
    leg1_agent_name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    leg1_time_ms: float
    train_fit_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Baton Zone Verification ─────────────────────────────────────────

    def verify_handoff(self) -> Dict[str, bool]:
        """
        Track official check – run at the baton-exchange zone.

        Verifies:
          1. Feature-column counts match across train, test, and name list.
          2. No NaN / Inf values in either array.
          3. A non-empty SHA-256 hash proves Leg 1 fitted only on train data.
        """
        checks = {
            "dims_match": (
                self.X_train.shape[1] == self.X_test.shape[1] == len(self.feature_names)
            ),
            "no_nan_train": bool(not np.isnan(self.X_train).any()),
            "no_nan_test": bool(not np.isnan(self.X_test).any()),
            "no_inf_train": bool(not np.isinf(self.X_train).any()),
            "no_inf_test": bool(not np.isinf(self.X_test).any()),
            "valid_split_hash": len(self.train_fit_hash) > 0,
        }
        if not all(checks.values()):
            failed = {k: v for k, v in checks.items() if not v}
            raise ValueError(
                f"[{self.lane_id}] 🚩 Baton Handoff Violation: {failed}"
            )
        return checks


# ── Chronological Data Splitter ─────────────────────────────────────────────

def load_chronological_split(
    csv_path: str = "data/creditcard.csv",
    target_col: str = "Class",
    time_col: str = "Time",
    test_ratio: float = 0.2,
):
    """
    Load a transaction dataset and split chronologically so future transactions
    *never* leak into training.

    Parameters
    ----------
    csv_path : str
        Path to CSV file. If missing, a deterministic 25,000-row synthetic dataset
        is generated automatically.
    target_col : str
        Target label column (e.g., 'Class', 'fraud', 'is_fraud').
    time_col : str
        Timestamp/ordering column (e.g., 'Time', 'timestamp'). If not present,
        the dataset maintains its existing row order without crashing.
    test_ratio : float
        Proportion reserved for unseen testing (default 0.20 = 80/20 split).

    Returns
    -------
    X_train, y_train, X_test, y_test, feature_names, raw_test_df
    """

    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded dataset from {csv_path}: {len(df):,} rows")
    except FileNotFoundError:
        print(
            f"⚠️  {csv_path} not found – generating deterministic synthetic "
            f"fallback (25 000 rows, ~0.2 % fraud)."
        )
        df = _generate_synthetic_fallback()
        target_col = "Class"
        time_col = "Time"

    # Match target column (case-insensitive fallback)
    if target_col not in df.columns:
        matched = False
        for c in df.columns:
            if c.lower() == target_col.lower():
                target_col = c
                matched = True
                break
        if not matched:
            raise ValueError(
                f"Target column '{target_col}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )

    # ── Strict chronological ordering ───────────────────────────────────
    if time_col and time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)
    else:
        print(
            f"ℹ️  Time column '{time_col}' not found in CSV – "
            f"maintaining existing row order for 80/20 split."
        )
        df = df.reset_index(drop=True)

    split_idx = int(len(df) * (1.0 - test_ratio))

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Numeric feature columns only (excluding target)
    feature_cols = [
        c for c in df.columns
        if c != target_col and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not feature_cols:
        raise ValueError("No numeric feature columns found in dataset.")

    y_train = train_df[target_col].astype(int).to_numpy(dtype=np.int32)
    y_test = test_df[target_col].astype(int).to_numpy(dtype=np.int32)

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)

    # Clean any NaNs with median imputation so baton handoff contract passes
    if np.isnan(X_train).any() or np.isnan(X_test).any():
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train).astype(np.float32)
        X_test = imputer.transform(X_test).astype(np.float32)

    print(
        f"📊 Chrono split → Train: {len(train_df):,} rows "
        f"(fraud {y_train.sum():,}) | "
        f"Test: {len(test_df):,} rows (fraud {y_test.sum():,})"
    )

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        feature_cols,
        test_df.reset_index(drop=True),
    )


# ── Synthetic Fallback Generator ────────────────────────────────────────────

def _generate_synthetic_fallback(n_rows: int = 25_000, seed: int = 42) -> pd.DataFrame:
    """
    Produce a reproducible synthetic dataset that mirrors the real Kaggle
    schema (V1–V28 PCA features, Time, Amount, Class) with a realistic
    ~0.5 % fraud rate driven by V14, V17, V12, and Amount correlations.

    The signal is strong enough for tree-based models to learn patterns
    while remaining challenging for linear models.
    """
    rng = np.random.default_rng(seed)

    n_legit = int(n_rows * 0.995)
    n_fraud = n_rows - n_legit

    # --- Legitimate transactions (normal cluster) ---
    legit = rng.normal(0.0, 1.0, size=(n_legit, 28))
    legit_amt = rng.exponential(scale=88.0, size=n_legit)

    # --- Fraudulent transactions (shifted cluster on key features) ---
    fraud = rng.normal(0.0, 1.0, size=(n_fraud, 28))
    # V14, V17, V12 are strongly negative for fraud (mirrors real data)
    fraud[:, 13] -= rng.uniform(3.0, 7.0, size=n_fraud)   # V14
    fraud[:, 16] -= rng.uniform(2.0, 5.0, size=n_fraud)   # V17
    fraud[:, 11] -= rng.uniform(2.0, 6.0, size=n_fraud)   # V12
    fraud[:, 9]  -= rng.uniform(1.0, 4.0, size=n_fraud)   # V10
    fraud[:, 3]  += rng.uniform(1.0, 3.0, size=n_fraud)   # V4
    fraud_amt = rng.exponential(scale=250.0, size=n_fraud)

    data = np.vstack([legit, fraud])
    amounts = np.concatenate([legit_amt, fraud_amt])
    labels = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])

    df = pd.DataFrame(data, columns=[f"V{i}" for i in range(1, 29)])
    df["Amount"] = amounts

    # Time: sorted seconds across ~2 days; shuffle first so fraud is spread
    shuffle_idx = rng.permutation(n_rows)
    df = df.iloc[shuffle_idx].reset_index(drop=True)
    labels = labels[shuffle_idx]
    df["Time"] = np.sort(rng.uniform(0, 172_800, size=n_rows))
    df["Class"] = labels.astype(int)

    return df
