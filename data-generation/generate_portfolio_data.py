"""Generate reproducible synthetic monthly portfolio data for Week 5.

Run from any directory with: python3 path/to/generate_portfolio_data.py
The generated CSV is written to the project's data directory.

All observations and relationships are simulated.  The series are designed for
teaching: past returns and a contemporaneously observable stress index contain
some, but far from perfect, information about next month's returns.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ASSETS = ["global_equity", "government_bonds", "corporate_bonds", "gold"]


def generate_portfolio_data(n_months=192, seed=42):
    """Return synthetic monthly returns and a market-stress indicator."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-31", periods=n_months, freq="ME")

    # Stress is observable at each month end and remains between 0 and 1.
    # It is persistent, so calm and stressed periods tend to last several months.
    stress = np.empty(n_months)
    stress[0] = 0.45
    for month in range(1, n_months):
        stress[month] = np.clip(
            0.15 + 0.68 * stress[month - 1] + rng.normal(0, 0.10),
            0.02,
            0.98,
        )

    # Monthly means and volatilities are expressed as decimal returns.
    base_returns = np.array([0.0075, 0.0025, 0.0040, 0.0035])
    volatilities = np.array([0.038, 0.012, 0.019, 0.029])

    # Equity and corporate bonds tend to move together. Government bonds and
    # gold provide some diversification in this deliberately simplified market.
    correlations = np.array([
        [1.00, -0.20, 0.55, -0.05],
        [-0.20, 1.00, 0.05, 0.12],
        [0.55, 0.05, 1.00, 0.04],
        [-0.05, 0.12, 0.04, 1.00],
    ])
    covariance = np.outer(volatilities, volatilities) * correlations

    # Different assets respond differently to stress. Small momentum effects
    # create a modest role for the previous month's returns.
    stress_effect = np.array([-0.040, 0.014, -0.016, 0.024])
    momentum = np.array([0.16, -0.04, 0.10, 0.04])

    returns = np.empty((n_months, len(ASSETS)))
    returns[0] = base_returns + rng.multivariate_normal(
        np.zeros(len(ASSETS)), covariance
    )

    for month in range(1, n_months):
        conditional_mean = (
            base_returns
            + stress_effect * (stress[month - 1] - 0.45)
            + momentum * returns[month - 1]
        )
        returns[month] = conditional_mean + rng.multivariate_normal(
            np.zeros(len(ASSETS)), covariance
        )

    # Guard against implausibly extreme observations while retaining bad months.
    returns = np.clip(returns, -0.18, 0.18)

    data = pd.DataFrame(returns, columns=ASSETS)
    data.insert(0, "stress_index", np.round(stress, 4))
    data.insert(0, "date", dates.strftime("%Y-%m-%d"))

    # Six decimals preserve the generated values without making the CSV noisy.
    data[ASSETS] = data[ASSETS].round(6)
    return data


def main():
    output = Path(__file__).resolve().parent.parent / "data" / "portfolio_monthly.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    data = generate_portfolio_data()
    data.to_csv(output, index=False)

    print(f"Wrote {len(data)} synthetic months to {output}")
    print(f"Date range: {data['date'].iloc[0]} to {data['date'].iloc[-1]}")
    print("Average monthly returns:")
    print(data[ASSETS].mean().round(4).to_string())


if __name__ == "__main__":
    main()
