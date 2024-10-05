import pandas as pd
from datetime import timedelta

def make_forecast(df: pd.DataFrame):
    # Create a new column for the forecast by shifting the Production_MWh column
    df = df.copy()  # Create a copy to avoid the SettingWithCopyWarning
    df.loc[:, 'Forecast_MWh'] = df['Production_MWh'].shift(1)

    # Remove rows where there is no previous data to forecast
    df = df.dropna(subset=['Forecast_MWh'])

    # Create a new column for the forecast hour by shifting the HourDK column by 1 day
    df.loc[:, 'Forecast_HourDK'] = pd.to_datetime(df['HourDK']) + timedelta(days=1)

    # Select only the relevant columns for the forecast: Forecast_HourDK and Forecast_MWh
    forecast_df = df[['Forecast_HourDK', 'Forecast_MWh']].reset_index(drop=True)

    return forecast_df
