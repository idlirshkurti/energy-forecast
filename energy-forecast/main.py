import pandas as pd
from data_loader import fetch_energy_data
from forecast import make_forecast
from datetime import datetime, timedelta

def save_forecast(df: pd.DataFrame):
    today = datetime.now() - timedelta(days=9)
    df.to_csv(f"forecast_data/forecast_{today.strftime('%Y_%m_%d')}.csv", index=False)

if __name__ == "__main__":
    # Load the data
    df = fetch_energy_data()

    if df is not None:
        # Make the forecast
        forecast_df = make_forecast(df)

        # Save the forecast
        save_forecast(forecast_df)
