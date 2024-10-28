## 24-Hour Energy Production Forecast with Conformal Prediction Intervals

![Streamlit dashboard with conformal predictions](./plots/Screenshot%202024-10-28%20at%2012.58.02.png)

This Streamlit module provides a tool for forecasting solar energy production in Denmark's DK1 price area. The app utilizes an LSTM model to predict the next 24 hours of energy production based on the most recent 72 hours of historical data. Conformal prediction intervals are applied to account for uncertainty in the forecasts, providing a user-defined confidence interval around each prediction.

### Features
- **24-Hour Energy Forecast**: Predicts solar energy production for the next 24 hours based on recent historical data.
- **Uncertainty Bounds**: Uses conformal prediction intervals to provide upper and lower bounds on forecasts, offering a confidence interval for each forecasted point.
- **Visualization**: Plots the last 72 hours of production along with the 24-hour forecast and prediction intervals.

### Setup and Installation
1. **Clone the Repository**: Clone this repository to your local machine.
    ```bash
    git clone https://github.com/your-username/energy-forecast.git
    cd energy-forecast
    ```

2. **Install Dependencies**: Ensure that you have Python 3.7 or later and install dependencies using `pip`.
    ```bash
    poetry install
    ```

3. **Add Model File**: Place the pre-trained LSTM model file (`solar_energy_production_model.keras`) in the `models/` directory.

4. **Run the Streamlit App**:
    ```bash
    poetry shell
    streamlit run app.py
    ```

### Usage
1. **Select Forecast Date**: Choose the date for which you’d like to generate a 24-hour forecast.
2. **Set Confidence Level**: Use the slider to select the desired confidence level (e.g., 95%) for the conformal prediction interval.
3. **Generate Forecast**: Click the "Generate Forecast" button. The app will:
   - Fetch the necessary historical data,
   - Calculate residuals for conformal prediction intervals,
   - Generate and display a 24-hour forecast with uncertainty bounds.
4. **View Results**: The app plots the last 72 hours of historical production, the forecasted values, and shaded prediction intervals to visualize uncertainty.

### API and Data
- **Data Source**: The app fetches hourly energy production data from [Energinet's Energi Data Service API](https://api.energidataservice.dk/).
- **Forecast Method**: Forecasts are based on a Long Short-Term Memory (LSTM) model trained to predict solar energy production. The model uses recent hourly production data to predict the next 24 hours of production.
- **Conformal Prediction**: Conformal prediction intervals are generated based on residual errors from past forecasts, providing uncertainty bounds for each forecasted value.

### Notes
- **Data Availability**: The app depends on Energinet’s data availability. Ensure the selected forecast date has sufficient historical data for accurate predictions.
- **Calibration Period**: The app retrieves 30 days of historical data by default for conformal prediction calibration. Adjust this if necessary.

---

This section should help users understand the purpose and functionality of the Streamlit app, guiding them through setup, usage, and data requirements. Let me know if there’s anything specific you’d like added or adjusted!
