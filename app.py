"""
This script contains the Streamlit app for generating 24-hour energy production forecasts with conformal prediction intervals.

The app allows the user to select a forecast date and confidence level for the prediction intervals. 
It fetches historical energy production data, calibrates the model, and generates the forecast with prediction intervals.

Project Structure:
- models/: Contains the pre-trained model for energy production forecasting.
- energy-forecast/: Contains the main code files for the energy production forecasting project.
- app.py: Streamlit app for generating energy production forecasts.
- pyproject.toml: Poetry configuration file.

To run the app, use the following command:
streamlit run app.py
"""

import streamlit as st
import requests
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('./models/solar_energy_production_model.keras')

def fetch_energy_data(start_date: datetime.datetime.date, end_date:datetime.datetime.date):
    """
    Fetches the energy data from the API for the given date range.
    Args:
        start_date: The start date of the date range.
        end_date: The end date of the date range.

    Returns:
        A pandas dataframe containing the energy data.
    """
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')

    url = f"https://api.energidataservice.dk/dataset/DeclarationProduction?start={start_date}&end={end_date}&filter=%7B%22PriceArea%22%3A%5B%22DK1%22%5D%7D"
    response = requests.get(url)
    data = response.json()

    if 'records' in data and data['records']:
        df = pd.DataFrame(data['records'])
        df = df[df['ProductionType'] == 'Solar']
        df = df[['HourDK', 'Production_MWh']]
        df = df.groupby('HourDK').sum().reset_index()
        return df
    else:
        st.warning("No data available for the given date range.")
        return pd.DataFrame()  # Return an empty DataFrame for consistency

def get_calibration_data():
    """
    Fetches and preprocesses calibration data for calculating residuals.
    """
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=30)  # Fetch 30 days of data for calibration
    
    # Fetch data
    df = fetch_energy_data(start_date=start_date, end_date=end_date)

    if df is None or df.empty:
        st.error("No calibration data available for the specified date range.")
        return None

    # Convert 'HourDK' to datetime and set as index
    df['HourDK'] = pd.to_datetime(df['HourDK'])
    df.set_index('HourDK', inplace=True)

    # Check if there is enough data for conformal prediction calibration
    if len(df) < 96:  # 72 hours for sequence + 24 hours for forecast horizon
        st.error("Insufficient data in the calibration range. Try extending the calibration range further.")
        return None

    return df


# Function to calculate residuals for conformal prediction
def calculate_residuals_for_conformal(model, calibration_data, sequence_length=72, forecast_horizon=24):
    """
    Calculate residuals between predictions and actuals in calibration data.
    """
    if len(calibration_data) < sequence_length + forecast_horizon:
        st.warning("Insufficient calibration data to calculate residuals. Try extending the calibration date range.")
        return np.array([])  # Return an empty array if there's insufficient data

    scaler = MinMaxScaler()
    calibration_data['Scaled_Production'] = scaler.fit_transform(calibration_data[['Production_MWh']])
    residuals = []

    # Loop through calibration data, checking if there's enough data for each sequence
    for start in range(len(calibration_data) - sequence_length - forecast_horizon):
        # Prepare input sequence
        input_sequence = calibration_data['Scaled_Production'].values[start : start + sequence_length]
        input_sequence = input_sequence.reshape((1, sequence_length, 1))

        # Predict and inverse scale
        scaled_forecast = model.predict(input_sequence)
        forecast = scaler.inverse_transform(scaled_forecast).flatten()
        forecast = np.clip(forecast, 0, None)

        # Retrieve actual values
        actual_values = calibration_data['Production_MWh'].values[start + sequence_length : start + sequence_length + forecast_horizon]

        # Calculate residuals (absolute error)
        residuals.extend(np.abs(forecast - actual_values))

    # Convert residuals to an array
    residuals = np.array(residuals)
    if residuals.size == 0:
        st.error("No residuals were calculated. Please check the calibration data range or model configuration.")
    return residuals


def get_historical_data_for_forecast(end_date, sequence_length=72):
    start_date = end_date - datetime.timedelta(hours=sequence_length)
    df = fetch_energy_data(start_date=start_date, end_date=end_date)
    
    if df.empty:
        st.error("Insufficient historical data to make the forecast.")
        return None

    df['HourDK'] = pd.to_datetime(df['HourDK'])
    df.set_index('HourDK', inplace=True)

    if len(df) < sequence_length:
        st.error("Not enough historical data points available.")
        return None

    return df

def make_conformal_forecast(model, end_date, residuals, sequence_length=72, forecast_horizon=24, alpha=0.05):
    historical_data = get_historical_data_for_forecast(end_date, sequence_length=sequence_length)
    if historical_data is None:
        return None, None

    scaler = MinMaxScaler()
    historical_data['Scaled_Production'] = scaler.fit_transform(historical_data[['Production_MWh']])

    last_sequence = historical_data['Scaled_Production'].values[-sequence_length:]
    last_sequence = last_sequence.reshape((1, sequence_length, 1))

    scaled_forecast = model.predict(last_sequence)
    forecast = scaler.inverse_transform(scaled_forecast).flatten()
    forecast = np.clip(forecast, 0, None)

    if residuals.size == 0:
        st.error("Residuals array is empty, cannot calculate prediction intervals.")
        return None, None

    q_upper = np.quantile(residuals, 1 - alpha / 2)
    lower_bound = forecast - q_upper
    upper_bound = forecast + q_upper
    lower_bound = np.clip(lower_bound, 0, None)

    forecast_index = pd.date_range(historical_data.index[-1] + pd.Timedelta(hours=1), periods=forecast_horizon, freq='H')
    forecast_df = pd.DataFrame({
        'Forecasted_Production_MWh': forecast,
        'Lower_Bound': lower_bound,
        'Upper_Bound': upper_bound
    }, index=forecast_index)

    return forecast_df, historical_data

# Streamlit app layout
st.title("24-Hour Energy Production Forecast with Conformal Prediction Intervals")

forecast_date = st.date_input("Select Forecast Date", datetime.date.today())
confidence_level = st.slider("Confidence Level", 0.80, 0.99, 0.95)

if st.button("Generate Forecast"):
    with st.spinner("Fetching data and generating forecast..."):
        calibration_data = get_calibration_data()
        if calibration_data is not None:
            residuals = calculate_residuals_for_conformal(model, calibration_data)
            
            if residuals.size > 0:
                alpha = 1 - confidence_level
                forecast_df, historical_data = make_conformal_forecast(model, forecast_date, residuals, alpha=alpha)

                if forecast_df is not None:
                    st.subheader("Forecast Plot")

                    fig, ax = plt.subplots(figsize=(14, 7))
                    ax.plot(historical_data.index[-72:], historical_data['Production_MWh'].values[-72:], label='Last 72 Hours of Production', color='blue')
                    ax.plot(forecast_df.index, forecast_df['Forecasted_Production_MWh'], label='24-Hour Forecast', color='red', linestyle='--')
                    ax.fill_between(forecast_df.index, forecast_df['Lower_Bound'], forecast_df['Upper_Bound'], color='pink', alpha=0.3, label=f'{int(confidence_level * 100)}% Prediction Interval')
                    ax.axvline(x=forecast_df.index[0], color='gray', linestyle='--', label='Forecast Start')

                    ax.set_xlabel('Time')
                    ax.set_ylabel('Production (MWh)')
                    ax.set_title('Energy Production Forecast with Prediction Intervals')
                    ax.legend()

                    st.pyplot(fig)
                else:
                    st.error("Forecast could not be generated due to missing data.")
            else:
                st.error("Residuals calculation returned empty, cannot generate prediction intervals.")
        else:
            st.error("Failed to retrieve or process data.")
