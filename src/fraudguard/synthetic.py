"""Utilities to generate reproducible synthetic transactions for tests and development."""

import numpy as np
import pandas as pd


def generate_synthetic_transactions(
    n_rows: int = 2000,
    seed: int = 42,
    fraud_rate: float = 0.035,
    start_time: int = 86400,
    days: float = 30.0,
    include_target: bool = True,
) -> pd.DataFrame:
    """Generate synthetic transactions with the exact 12 allowlisted predictors and multi-day temporal span."""
    rng = np.random.default_rng(seed)

    # Identifiers and timestamps
    transaction_ids = np.arange(1000000, 1000000 + n_rows)
    # Time steps spanning the requested days
    total_seconds = int(days * 86400)
    avg_step = max(1, int(total_seconds / max(1, n_rows)))
    time_deltas = rng.integers(0, max(2, 2 * avg_step), size=n_rows)
    time_deltas[rng.random(n_rows) < 0.15] = 0  # 15% timestamp ties
    transaction_dt = start_time + np.cumsum(time_deltas)

    # 12 Allowlisted Predictors
    # Numeric
    transaction_amt = np.round(rng.lognormal(mean=4.2, sigma=1.1, size=n_rows), 2)
    transaction_amt = np.clip(transaction_amt, 1.0, 10000.0)

    # Nullable numerics
    dist1 = rng.exponential(scale=25.0, size=n_rows)
    dist1[rng.random(n_rows) < 0.60] = np.nan  # 60% missing
    dist1 = np.round(dist1, 1)

    dist2 = rng.exponential(scale=40.0, size=n_rows)
    dist2[rng.random(n_rows) < 0.85] = np.nan  # 85% missing
    dist2 = np.round(dist2, 1)

    # Categoricals
    product_cds = rng.choice(
        ["W", "C", "R", "H", "S"], size=n_rows, p=[0.70, 0.12, 0.08, 0.06, 0.04]
    )

    card4_choices = ["visa", "mastercard", "discover", "american express", None]
    card4 = rng.choice(card4_choices, size=n_rows, p=[0.65, 0.28, 0.04, 0.02, 0.01])

    card6_choices = ["debit", "credit", "debit or credit", "charge card", None]
    card6 = rng.choice(card6_choices, size=n_rows, p=[0.74, 0.24, 0.01, 0.005, 0.005])

    addr1_codes = [299, 325, 204, 315, 126, 441, 181, 272, None]
    addr1_p = [0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.07, 0.05, 0.10]
    addr1 = rng.choice(addr1_codes, size=n_rows, p=addr1_p)

    addr2_codes = [87, 60, 96, None]
    addr2 = rng.choice(addr2_codes, size=n_rows, p=[0.90, 0.04, 0.01, 0.05])

    p_emails = [
        "gmail.com",
        "yahoo.com",
        "hotmail.com",
        "anonymous.com",
        "aol.com",
        "comcast.net",
        None,
    ]
    p_email_p = [0.40, 0.20, 0.15, 0.08, 0.05, 0.02, 0.10]
    p_emaildomain = rng.choice(p_emails, size=n_rows, p=p_email_p)

    r_emails = ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", None]
    r_email_p = [0.10, 0.05, 0.03, 0.02, 0.80]  # 80% missing
    r_emaildomain = rng.choice(r_emails, size=n_rows, p=r_email_p)

    m4_choices = ["M0", "M1", "M2", None]
    m4 = rng.choice(m4_choices, size=n_rows, p=[0.40, 0.25, 0.15, 0.20])

    m6_choices = ["T", "F", None]
    m6 = rng.choice(m6_choices, size=n_rows, p=[0.50, 0.35, 0.15])

    data = {
        "TransactionID": transaction_ids,
        "TransactionDT": transaction_dt,
        "TransactionAmt": transaction_amt,
        "ProductCD": product_cds,
        "dist1": dist1,
        "dist2": dist2,
        "card4": card4,
        "card6": card6,
        "addr1": addr1,
        "addr2": addr2,
        "P_emaildomain": p_emaildomain,
        "R_emaildomain": r_emaildomain,
        "M4": m4,
        "M6": m6,
    }

    if include_target:
        # Synthetic risk signal correlated with amount, ProductCD=C, credit cards, night hours, anonymous emails
        hour_of_day = (transaction_dt % 86400) / 3600.0
        night_risk = ((hour_of_day >= 1.0) & (hour_of_day <= 5.0)).astype(float)

        risk_score = (
            (transaction_amt > 200.0).astype(float) * 1.6
            + (product_cds == "C").astype(float) * 2.2
            + (product_cds == "R").astype(float) * 1.4
            + (card6 == "credit").astype(float) * 1.1
            + (p_emaildomain == "anonymous.com").astype(float) * 2.0
            + (pd.isna(addr1)).astype(float) * 1.2
            + night_risk * 1.5
            + ((transaction_amt > 500.0) & (card4 == "discover")).astype(float) * 1.8
            + rng.normal(0, 1.2, size=n_rows)
        )
        threshold = np.quantile(risk_score, 1.0 - fraud_rate)
        is_fraud = (risk_score >= threshold).astype(int)
        data["isFraud"] = is_fraud

    df = pd.DataFrame(data)
    # Sort chronologically
    df = df.sort_values(by=["TransactionDT", "TransactionID"]).reset_index(drop=True)
    return df
