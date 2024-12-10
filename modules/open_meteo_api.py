from datetime import date, datetime
import os
import json
import requests
import csv
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd


def open_meteo_request(
    start_date: str,
    end_date: str,
    latitude: float,
    longitude: float,
    hourly: Optional[List[str]],
    base_url: str = "https://archive-api.open-meteo.com/v1/archive",
) -> Optional[Dict[str, Any]]:

    params = {
        'latitude': latitude,
        'longitude': longitude,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': ','.join(hourly),
        'windspeed_unit': 'mph',
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        try:
            result = response.json()
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response: {e}")
            return None

        if 'hourly' not in result:
            print("No 'hourly' data found in the response.")
            return None

        data = result['hourly']
        return data

    except requests.RequestException as e:
        print(f"HTTP Error: {e}")
        return None