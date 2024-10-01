import os
import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_energy_data():
    yesterday = datetime.now() - timedelta(days=10)
    today = datetime.now() - timedelta(days=9)

    start_date = yesterday.strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')

    url = f"https://api.energidataservice.dk/dataset/DeclarationProduction?start={start_date}&end={end_date}&filter=%7B%22PriceArea%22%3A%5B%22DK1%22%5D%7D"
    response = requests.get(url)
    data = response.json()

    # Process and structure the data
    if data['records']:
        df = pd.DataFrame(data['records'])
        df.to_csv(f"/energy_forecast/forecast_{today.strftime('%Y_%m_%d')}.csv", index=False)
    else:
        print("No data available for the given date range.")

    print("CSV file created:", os.path.exists(f"energy_forecast/forecast_{today.strftime('%Y_%m_%d')}.csv"))

if __name__ == "__main__":
    fetch_energy_data()
