from preprocessing import load_and_prepare
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

df_min, X, _, X_scaled = load_and_prepare()

model = KMeans(n_clusters=3, random_state=42)
df_min['cluster'] = model.fit_predict(X_scaled)

df_min.to_csv('results/clustering.csv')

# Pairplot
sns.pairplot(df_min.reset_index(), hue='cluster',
             vars=['potencia', 'ESP32_temp', 'WORKSTATION_CPU'])
plt.suptitle("Clustered Modes")
plt.savefig('results/clustering_plot.png')
