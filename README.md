# Energy Forecast

This repository contains a Python script that fetches hourly energy production data for the DK1 price area from the Energi Data Service API. The script runs daily via GitHub Actions, using the last 24 hours of data as a forecast for the next 24 hours. Forecasts are saved as CSV files in the repository.

## Features

- Daily API calls to retrieve energy production data
- Hourly forecasts based on the latest data
- CSV output for easy data access

## Setup

Ensure you have the necesary requirements installed as shown in `pyproject.toml`

## Usage

The script `main.py` will run automatically through GitHub Actions, scheduled to execute at midnight UTC.

## License

This project is licensed under the MIT License.

Feel free to modify it according to your specific needs or add more details as necessary!
