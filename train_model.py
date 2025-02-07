# train_model.py
import pandas as pd
from xgboost import XGBRegressor
import joblib

# Generate synthetic training data
np.random.seed(42)
data = {
    "distance_km": np.random.uniform(1e6, 5e6, 1000),
    "payload_kg": np.random.randint(1000, 5000, 1000),
    "engine_type": np.random.choice([0, 1, 2], 1000),
    "gravity_force": np.random.uniform(3.7, 9.8, 1000),
    "fuel_used_kg": np.random.uniform(5000, 20000, 1000)
}
df = pd.DataFrame(data)

model = XGBRegressor()
model.fit(df.drop("fuel_used_kg", axis=1), df["fuel_used_kg"])
joblib.dump(model, "fuel_model.pkl")