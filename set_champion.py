# set_champion.py - OPTIONAL shortcut for the MLflow UI step (Lab 1 Lab 6 / Lab 2 C1).
# Finds the registered version with the LOWEST mae (newest wins a tie)
# and gives it the alias "champion".

import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()
NAME = "car-price-model"

best = None
for v in client.search_model_versions(f"name='{NAME}'"):
    mae = client.get_run(v.run_id).data.metrics.get("mae")
    if mae is None:
        continue
    print(f"version {v.version}: mae = {mae:.3f}")
    if best is None or mae < best[1] or (mae == best[1] and int(v.version) > int(best[0])):
        best = (v.version, mae)

client.set_registered_model_alias(NAME, "champion", best[0])
print(f"\n'champion' now points to version {best[0]} (mae {best[1]:.3f})")
