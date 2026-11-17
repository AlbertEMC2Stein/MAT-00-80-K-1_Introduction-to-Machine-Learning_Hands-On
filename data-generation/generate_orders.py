"""Generate a reproducible synthetic packing-time classification dataset.

Run from any directory with: python3 path/to/generate_orders.py
The generated CSV is written to the project's data directory.
All observations and relationships are simulated, not real business data.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_orders(n_orders=2000, seed=42):
    """Return synthetic orders with observed packing times."""
    rng = np.random.default_rng(seed)
    item_count = rng.integers(1, 21, size=n_orders)
    fragile_count = rng.binomial(item_count, 0.2)
    # Sample uniformly from 1 through min(item_count, 4), inclusive.
    storage_zones = rng.integers(1, np.minimum(item_count, 4) + 1)
    gift_wrap = rng.binomial(1, 0.2, size=n_orders)

    fragile_weight = rng.uniform(0.2, 1.5, size=n_orders)
    regular_weight = rng.uniform(0.1, 3.0, size=n_orders)
    total_weight = np.round(
        fragile_count * fragile_weight
        + (item_count - fragile_count) * regular_weight,
        2,
    )
    # The planned package is a cube; its side is known before packing.
    package_side = np.round(np.clip(
        15 + 1.2 * item_count + 0.3 * total_weight
        + rng.normal(0, 4, size=n_orders), 15, 55
    ), 2)
    avg_item_price = rng.lognormal(mean=2.5, sigma=0.5, size=n_orders)
    order_value = np.round(
        (item_count + 1.5 * fragile_count) * avg_item_price + 5 * gift_wrap, 2
    )

    expected_time = (
        3 + 0.35 * item_count + 0.60 * fragile_count
        + 0.25 * total_weight + 1.5 * gift_wrap
        + 0.04 * item_count * storage_zones
        + 0.0001 * package_side ** 3
    )
    packing_time = expected_time + rng.normal(0, 2, size=n_orders)
    # Ensure durations remain positive after rounding to hundredths of a minute.
    invalid = packing_time < 0.005
    while invalid.any():
        packing_time[invalid] = expected_time[invalid] + rng.normal(0, 2, invalid.sum())
        invalid = packing_time < 0.005

    packing_time = np.round(packing_time, 2)
    return pd.DataFrame({
        "order_id": np.arange(1, n_orders + 1),
        "item_count": item_count,
        "fragile_count": fragile_count,
        "storage_zones": storage_zones,
        "gift_wrap": gift_wrap,
        "total_weight_kg": total_weight,
        "order_value_eur": order_value,
        "package_side_cm": package_side,
        "packing_time_min": packing_time,
    })


def main():
    output = Path(__file__).resolve().parent.parent / "data" / "orders.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    orders = generate_orders()
    orders.to_csv(output, index=False)
    print(f"Wrote {len(orders)} synthetic orders to {output}")
    print("Packzeiten über 15 Minuten:", (orders["packing_time_min"] > 15).sum())


if __name__ == "__main__":
    main()
