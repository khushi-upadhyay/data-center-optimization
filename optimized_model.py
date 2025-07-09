import pandas as pd
import numpy as np
import xgboost as xgb   
import matplotlib.pyplot as plt


df = pd.read_csv("cleaned_dataset_lstm.csv", index_col="fecha_servidor", parse_dates=True)


xgb_model = xgb.Booster()
xgb_model.load_model("models/best_xgboost_model.json")

def predict_energy(X):
    # Drop 'weekday'
    if 'weekday' in X.columns:
        X = X.drop(columns=['weekday'])
    dmat = xgb.DMatrix(X)
    return xgb_model.predict(dmat)


X = df.drop(columns=["energia"])
y_pred_base = predict_energy(X)

def report_savings(y_base, y_new, label):
    saved_pct = 100 * (y_base - y_new) / y_base
    print(f"{label:30s} → avg saving: {saved_pct.mean():.2f}%")
    return saved_pct

#  Step 1: Reschedule to off‑peak hours (shift 9–17 to 2 AM)
X1 = X.copy()
mask = X1['hour'].between(9, 17)
X1.loc[mask, 'hour'] = 2
y1 = predict_energy(X1)
report_savings(y_pred_base, y1, "Shift peak→2 AM")

# Step 2: CPU Throttling (−20%)
X2 = X.copy()
X2['WORKSTATION_CPU'] *= 0.8
y2 = predict_energy(X2)
report_savings(y_pred_base, y2, "CPU usage −20%")

#  Step 3: Power‑Factor Improvement (+10% fp)
X3 = X.copy()
X3['fp'] = np.minimum(1.0, X3['fp'] * 1.1)
y3 = predict_energy(X3)
report_savings(y_pred_base, y3, "Power factor +10%")

#  Step 4: Dynamic Scaling Control (conditional CPU throttle)

threshold = np.percentile(y_pred_base, 75)
X4 = X.copy()


if 'weekday' in X4.columns:
    X4 = X4.drop(columns=['weekday'])


preds = predict_energy(X4)


throttle_mask = preds > threshold
X4.loc[throttle_mask, 'WORKSTATION_CPU'] *= 0.8


y4 = predict_energy(X4)
report_savings(y_pred_base, y4, "Dynamic CPU throttle (vectorized)")



plt.figure(figsize=(12, 4))
plt.plot(y_pred_base[:500], label="Baseline")
plt.plot(y2[:500], label="CPU−20%")
plt.legend()
plt.title("Energy: baseline vs CPU throttle")
plt.ylabel("Predicted Energia")
plt.tight_layout()
plt.show()
