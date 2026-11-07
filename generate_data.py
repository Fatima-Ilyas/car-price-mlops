# generate_data.py - creates the used-car datasets (run once; the CSVs are already included)
#
# Price rule (lakh PKR) the model must discover:
#   price = 0.10 * engine_cc - 6 * age_years - 0.12 * mileage_thousand_km + 40  (+ noise)
# Luxury imports sell ~5% above this rule - a brand premium the model has never seen.
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

def make(n, age, mileage, engine, noise=3, premium=1.0):
    m = n + 20                                              # a few extra rows, in case some are filtered out
    age_y = rng.integers(*age, size=m)
    km = np.round(age_y * rng.uniform(*mileage, size=m)).astype(int)   # older cars have more km
    cc = rng.choice(engine, size=m)
    price = (0.10 * cc - 6 * age_y - 0.12 * km + 40) * premium + rng.normal(0, noise, size=m)
    df = pd.DataFrame({"age_years": age_y, "mileage_thousand_km": np.maximum(km, 3),
                       "engine_cc": cc, "price_lakh": np.round(price).astype(int)})
    return df[df.price_lakh > 5].head(n)

engines = [660, 800, 1000, 1200, 1300, 1500, 1600, 1800, 2000]
normal = dict(age=(1, 13), mileage=(8, 20), engine=engines)
lux = dict(age=(1, 4), mileage=(3, 10), engine=[3000, 3500, 4000, 4500], noise=5, premium=1.05)

make(500, **normal).to_csv("cars.csv", index=False)                                  # Lab 1: first dataset (500 cars)
make(50, **normal).to_csv("data_batches/batch2_new_cars.csv", index=False)           # Lab 2 Part B: "November" data
make(30, **lux).to_csv("data_batches/batch3_luxury.csv", index=False)                # Lab 4: drift data (luxury imports)
make(20, **normal).to_csv("data_batches/batch4_extra_cars.csv", index=False)         # Lab 2 E13: extra batch
print("done")
