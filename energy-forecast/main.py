import os
import pandas as pd
import duckdb
from data_loader import fetch_energy_data
from forecast import make_forecast
from datetime import datetime, date, timedelta


def save_forecast_to_motherduck(df: pd.DataFrame):
    token = os.environ.get("MOTHERDUCK_TOKEN")
    if not token:
        raise ValueError("MOTHERDUCK_TOKEN environment variable is not set.")

    # Rename columns to match the MotherDuck table schema
    df = df.rename(columns={
        "Forecast_HourDK": "forecast_timestamp",
        "Forecast_MWh": "forecast_mwh",
    })
    df["forecast_run_date"] = date.today()
    df["price_area"] = "DK1"

    con = duckdb.connect(f"md:my_db?motherduck_token={token}")
    con.execute("""
        INSERT INTO solar_forecasts (forecast_run_date, forecast_timestamp, forecast_mwh, price_area)
        SELECT forecast_run_date, forecast_timestamp, forecast_mwh, price_area
        FROM df
    """)
    row_count = con.execute("SELECT COUNT(*) FROM solar_forecasts").fetchone()[0]
    con.close()
    print(f"Inserted {len(df)} rows. Total rows in solar_forecasts: {row_count}")


if __name__ == "__main__":
    # Load the data
    df = fetch_energy_data()

    if df is not None:
        # Make the forecast
        forecast_df = make_forecast(df)

        # Save the forecast to MotherDuck
        save_forecast_to_motherduck(forecast_df)
