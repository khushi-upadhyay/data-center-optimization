from preprocessing import load_and_prepare
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

df_min, X, _, X_scaled = load_and_prepare()

model = IsolationForest(contamination=0.01, random_state=42)
df_min['anomaly'] = model.fit_predict(X_scaled)
df_min['anomaly'] = df_min['anomaly'].map({1: 0, -1: 1})

df_min.to_csv('results/anomaly_detected.csv')

# Plot
plt.figure(figsize=(15,5))
plt.plot(df_min.index, df_min['energia'], label='Power')
plt.scatter(df_min[df_min['anomaly'] == 1].index,
            df_min[df_min['anomaly'] == 1]['potencia'],
            color='red', label='Anomaly')
plt.legend()
plt.savefig('results/anomalies_plot.png')
