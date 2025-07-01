import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_prepare(path='cleaned_dataset_lstm.csv', nrows=None):
    df = pd.read_csv(path, low_memory=False, encoding='utf-8', on_bad_lines='skip', nrows=nrows)

    # Convert to datetime
    df['fecha_servidor'] = pd.to_datetime(df['fecha_servidor'], errors='coerce')
    df = df.dropna(subset=['fecha_servidor'])  # Drop rows with bad timestamps
    df = df.set_index('fecha_servidor')

    # Drop non-numeric columns (e.g., MAC address, weekday, etc.)
    non_numeric_cols = ['MAC', 'weekday', 'fecha_esp32']
    df = df.drop(columns=non_numeric_cols, errors='ignore')

    # Resample every 1 minute (mean of each minute)
    df_min = df.resample('1min').mean().dropna()

    # Define features and target
    target = 'energia'  # Predicting power
    y = df_min[target]
    X = df_min.drop(columns=[target])

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df_min, X, y, X_scaled
