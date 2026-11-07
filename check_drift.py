# check_drift.py - are users asking about cars like the ones the model learned from?

import sys

import pandas as pd
from scipy.stats import ks_2samp

training = pd.read_csv("cars.csv")
live = pd.read_csv("predictions.csv").tail(50)   # the 50 most recent requests

if len(live) < 20:
    print("Not enough predictions yet - send some traffic first")
    sys.exit(0)

# p-value below this = "clearly different". 0.001 avoids false alarms on normal traffic.
ALPHA = 0.001

drift_found = False
for column in ["age_years", "mileage_thousand_km", "engine_cc"]:
    # KS test: how likely is it that both lists come from the same kind of cars?
    p_value = ks_2samp(training[column], live[column]).pvalue
    drifted = p_value < ALPHA
    drift_found = drift_found or drifted
    print(f"{column:20}  training avg {training[column].mean():7.0f}   "
          f"live avg {live[column].mean():7.0f}   p-value {p_value:.3f}   "
          f"{'DRIFT' if drifted else 'OK'}")

if drift_found:
    print("\nDRIFT DETECTED: users are asking about cars the model never learned from.")
    print("Action: collect real prices for these cars, add them to the data, retrain.")
    sys.exit(1)

print("\nNo drift: live cars look like the training cars.")
