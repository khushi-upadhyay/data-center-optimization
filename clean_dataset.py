import pandas as pd

path = "dataset.csv"
cleaned_path = "cleaned_dataset_lstm.csv"

try:
    df = pd.read_csv(
        path,
        engine="python",  # Fixes the EOF error
        on_bad_lines="skip",  # Skips corrupted lines
        encoding="utf-8"
        # Removed low_memory because it's not supported with engine="python"
    )
except Exception as e:
    print("❌ Error loading CSV:", e)
    exit()

# Drop unneeded columns
df.drop(columns=['MAC', 'fecha_esp32'], inplace=True, errors='ignore')

# Convert timestamp
df['fecha_servidor'] = pd.to_datetime(df['fecha_servidor'], errors='coerce')
df.dropna(subset=['fecha_servidor'], inplace=True)
df = df.set_index('fecha_servidor')

# Convert all columns to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Add time features for LSTM
df['hour'] = df.index.hour
df['day'] = df.index.day
df['weekday'] = df.index.weekday

# Drop rows with any missing values
df.dropna(inplace=True)

# Save cleaned dataset
df.to_csv(cleaned_path)
print(f"✅ Dataset cleaned and saved to {cleaned_path} — {len(df)} rows.")
