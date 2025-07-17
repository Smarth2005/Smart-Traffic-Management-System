import pandas as pd
from ml.traffic_predictor import TrafficFlowPredictor

def main():
    csv_file = "traffic_data_20250716.csv"

    try:
        df = pd.read_csv(csv_file)
        print("✅ Loaded CSV:", csv_file)
        print("📊 Initial shape:", df.shape)
        print("🧾 Columns:", list(df.columns))

        required_columns = [
            'weighted_total', 'average_speed', 'congestion_level',
            'traffic_flow_efficiency', 'average_waiting_time'
        ]

        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Keep only required columns
        features = df[required_columns].copy()
        # Fill missing average_speed with 0 or median
        features['average_speed'].fillna(0, inplace=True)

        df['target'] = df['weighted_total'].shift(-1).ffill()

        # Ensure no NaNs
        valid = features.notnull().all(axis=1) & df['target'].notnull()
        df = df.loc[valid, required_columns + ['target']]

        print("📉 NaNs in cleaned data:\n", df.isna().sum())
        print("✅ Final shape for training:", df.shape)

        if df.empty:
            raise ValueError("No valid rows left after filtering. Check data.")

        predictor = TrafficFlowPredictor()
        predictor.train(df)
        print(f"✅ Model trained and saved to {predictor.model_path}")

        metrics = predictor.evaluate(df)

    except Exception as e:
        print(f"❌ Training failed: {e}")

if __name__ == "__main__":
    main()
