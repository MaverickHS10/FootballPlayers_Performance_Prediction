import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import joblib

try:
    from xgboost import XGBRegressor 
    xgb_installed = True
except ImportError:
    xgb_installed = False
    print("XGBoost not installed. Skipping it from comparison.")

df = pd.read_csv("playersData.csv")

features = ['Min', 'xG', 'xAG', 'Sh', 'SoT', 'KP', 'PrgP', 'PrgC', 'SCA90', 'Age']
target = 'G+A'

df = df[features + [target]].dropna()

X = df[features]
y = df[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

results = []
model_objects = {}

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
results.append({
    "Model": "Linear Regression",
    "R2": r2_score(y_test, y_pred_lr),
    "RMSE": sqrt(mean_squared_error(y_test, y_pred_lr))
})
model_objects["Linear Regression"] = lr

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results.append({
    "Model": "Random Forest",
    "R2": r2_score(y_test, y_pred_rf),
    "RMSE": sqrt(mean_squared_error(y_test, y_pred_rf))
})
model_objects["Random Forest"] = rf

if xgb_installed:
    xgb = XGBRegressor(random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    results.append({
        "Model": "XGBoost",
        "R2": r2_score(y_test, y_pred_xgb),
        "RMSE": sqrt(mean_squared_error(y_test, y_pred_xgb))
    })
    model_objects["XGBoost"] = xgb

print("\n Model Comparison:")
print("{:<18} {:<10} {:<10}".format("Model", "R² Score", "RMSE"))
for res in results:
    print("{:<18} {:<10.3f} {:<10.2f}".format(res["Model"], res["R2"], res["RMSE"]))

best_model_result = min(results, key=lambda x: x["RMSE"])
best_model_name = best_model_result["Model"]
best_model = model_objects[best_model_name]

joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
with open("best_model_name.txt", "w") as f:
    f.write(best_model_name)

print(f"\n Best model saved: {best_model_name}")

#data visualization
model_names = [r["Model"] for r in results]
r2_scores = [r["R2"] for r in results]
rmses = [r["RMSE"] for r in results]

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.bar(model_names, r2_scores, color='skyblue')
plt.title("R² Scores")
plt.ylabel("R²")

plt.subplot(1, 2, 2)
plt.bar(model_names, rmses, color='salmon')
plt.title("RMSE")
plt.ylabel("RMSE")

plt.tight_layout()
plt.show()
