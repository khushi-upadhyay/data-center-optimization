from preprocessing import load_and_prepare
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import os

# Load data
df_min, X, y, X_scaled = load_and_prepare()

# Train model
model = LinearRegression()
model.fit(X_scaled, y)

# Predict
y_pred = model.predict(X_scaled)  # 🔥 This line must be included!

# Save metrics only
os.makedirs("results", exist_ok=True)

r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)

# Save metrics to CSV
metrics_df = pd.DataFrame({
    'Model': ['LinearRegression'],
    'R2': [r2],
    'RMSE': [rmse],
    'MAE': [mae]
})
metrics_df.to_csv('results/linear_metrics.csv', index=False)

# Show in console
print("R²:", r2)
print("RMSE:", rmse)
print("MAE:", mae)
