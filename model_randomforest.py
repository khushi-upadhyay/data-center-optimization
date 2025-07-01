from preprocessing import load_and_prepare
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd  # You missed importing pandas

df_min, X, y, _ = load_and_prepare()

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
y_pred = model.predict(X)

df_min['pred_rf'] = y_pred

# Calculate metrics
r2 = r2_score(y, y_pred)
rmse = mean_squared_error(y, y_pred, squared=False)  # RMSE
mae = mean_absolute_error(y, y_pred)

# Save metrics to CSV
metrics_df = pd.DataFrame({
    'Model': ['RandomForest'], 
    'R2': [r2],
    'RMSE': [rmse],
    'MAE': [mae]
})
metrics_df.to_csv('results/random_metrics.csv', index=False)

# Save feature importance plot
plt.figure(figsize=(8,6))
sns.barplot(x=model.feature_importances_, y=X.columns)
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig('results/rf_feature_importance.png')
plt.close()  # Close plot to avoid overlapping if run multiple times
