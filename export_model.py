# export_model.py - takes the "champion" model from MLflow and saves it as champion.pkl

import joblib
import mlflow
import mlflow.sklearn

# Use the same register (mlflow.db) that train.py writes to
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# "models:/<name>@<alias>" means: the version that has this alias
model = mlflow.sklearn.load_model("models:/car-price-model@champion")

joblib.dump(model, "champion.pkl")
print("Champion model saved to champion.pkl")
