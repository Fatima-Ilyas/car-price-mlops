# predict.py - uses the saved model to price a NEW car (no training here)

import pandas as pd
import joblib

# Load the model file that train.py created
model = joblib.load("model.pkl")

# A new car the model has never seen
new_car = pd.DataFrame({"age_years": [5], "mileage_thousand_km": [60], "engine_cc": [1300]})

price = model.predict(new_car)[0]
print(f"5 years old, 60k km, 1300 cc -> predicted price: {price:.1f} lakh")
