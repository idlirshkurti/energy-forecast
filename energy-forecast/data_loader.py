import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_energy_data():
    yesterday = datetime.now() - timedelta(days=15)
    today = datetime.now() - timedelta(days=14)

    start_date = yesterday.strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')

    url = f"https://api.energidataservice.dk/dataset/DeclarationProduction?start={start_date}&end={end_date}&filter=%7B%22PriceArea%22%3A%5B%22DK1%22%5D%7D"
    response = requests.get(url)
    data = response.json()

    # Process and structure the data
    if data['records']:
        df = pd.DataFrame(data['records'])
        df = df[df['ProductionType'] == 'Solar']
        df = df[['HourDK', 'Production_MWh']]
        df = df.groupby('HourDK').sum().reset_index()
        return df
    else:
        print("No data available for the given date range.")
        return None
