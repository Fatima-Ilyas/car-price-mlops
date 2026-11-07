# send_traffic.py - pretends to be 50 users calling the API
#
# Usage:  python send_traffic.py normal   -> cars like the ones the model learned from
#         python send_traffic.py drift    -> a new market: big-engine luxury imports

import random
import sys

import pandas as pd
import requests

# 127.0.0.1 instead of "localhost": on Windows, "localhost" adds ~2 seconds per request
URL = "http://127.0.0.1:8000/predict"
mode = sys.argv[1] if len(sys.argv) > 1 else "normal"
training = pd.read_csv("cars.csv")

for i in range(50):
    if mode == "drift":
        # Luxury imports: new, low-km, huge engines - nothing like the training data
        age = random.randint(1, 3)
        km = random.randint(3, 20)
        cc = random.randint(3000, 4500)
    else:
        # A car similar to a training car (km and cc changed by up to 5%)
        car = training.sample(1).iloc[0]
        age = int(car["age_years"])
        km = max(1, round(car["mileage_thousand_km"] * random.uniform(0.95, 1.05)))
        cc = round(car["engine_cc"] * random.uniform(0.95, 1.05))

    response = requests.post(URL, json={"age_years": age, "mileage_thousand_km": km, "engine_cc": cc})
    print(f"{age} yrs, {km}k km, {cc} cc -> {response.json()['price_lakh']} lakh")

print(f"\nSent 50 '{mode}' requests")
