from datetime import timedelta
import pickle
from fontTools.merge.util import current_time
from joblib import load
import pandas as pd
from modules.open_meteo_api import open_meteo_request
import datetime
import math
import pandas as pd
from datetime import timedelta
import requests
import logging


def add_weather_forecast(df):
    """
    Enriches the input DataFrame with weather-related features based on
    departure and arrival airport datetime information.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing flight details.

    Returns:
    - pd.DataFrame: DataFrame enriched with weather features.
    """


    # Define airport coordinates (Consider loading from a separate file or DataFrame)
    airport_coordinates = {
        "ATL": (33.6407, -84.4277),
        "DEN": (39.8561, -104.6737),
        "DFW": (32.8975, -97.0404),
        "IAH": (29.9902, -95.3368),
        "LAS": (36.0831, -115.1483),
        "LAX": (33.9428, -118.4100),
        "LGA": (40.7769, -73.8741),
        "ORD": (41.9742, -87.9073),
        "PHX": (33.4352, -112.0102),
        "SFO": (37.6213, -122.3790),
    }

    # Define feature mappings
    features = {
        "wind_speed_10m": "sped",
        "precipitation": "p01m",
        "visibility": "vsby",
        "temperature_2m": "tmpc",
    }

    # Define periods relative to scheduled times (in hours)
    periods = [-6, -1, 0, 1, 6]

    # Iterate over each flight record
    for index, row in df.iterrows():
        for i in range(2):
            port_type = 'origin' if i == 0 else 'destination'
            time_type = 'departure' if i == 0 else 'arrival'
            stat_name = 'DEPARTURE' if i == 0 else 'ARRIVAL'

            # Retrieve airport code and validate
            airport_code = row[f'{port_type}_airport']
            if airport_code not in airport_coordinates:
                logging.warning(f"Airport code '{airport_code}' not found in coordinates dictionary.")
                continue  # Skip to next iteration or handle accordingly

            lat, lon = airport_coordinates[airport_code]

            # Retrieve scheduled datetime and ensure it's a Timestamp
            scheduled_datetime = row[f'{time_type}_datetime']
            if not isinstance(scheduled_datetime, pd.Timestamp):
                scheduled_datetime = pd.to_datetime(scheduled_datetime)

            # Define the time window for weather data (1 day before and after)
            start_date = (scheduled_datetime - timedelta(days=1)).strftime('%Y-%m-%d')
            end_date = (scheduled_datetime + timedelta(days=1)).strftime('%Y-%m-%d')

            # Round scheduled time to the nearest hour
            scheduled_time_rounded = scheduled_datetime.replace(minute=0, second=0, microsecond=0)

            # Fetch weather data using the open_meteo_request function
            try:
                weather_dict = open_meteo_request(
                    start_date=start_date,
                    end_date=end_date,
                    latitude=lat,
                    longitude=lon,
                    hourly=["temperature_2m", "precipitation", "visibility", "wind_speed_10m"],
                    base_url="https://api.open-meteo.com/v1/forecast"
                )
            except Exception as e:
                logging.error(f"Error fetching weather data for {airport_code} at index {index}: {e}")
                continue  # Skip to next iteration or handle accordingly

            # Convert visibility from meters to miles
            weather_dict['visibility'] = [dist * 0.0006213712 for dist in weather_dict['visibility']]

            # Create DataFrame from weather data
            weather_df = pd.DataFrame(weather_dict)
            weather_df.rename(columns=features, inplace=True)
            weather_df['time'] = pd.to_datetime(weather_df['time'])

            # Iterate over defined periods to extract relevant weather features
            for p in periods:
                target_time = scheduled_time_rounded + timedelta(hours=p)

                # Fetch weather data for the target_time
                temp_weather = weather_df[weather_df['time'] == target_time]

                if temp_weather.empty:
                    logging.warning(f"No weather data available for {airport_code} at {target_time}.")
                    # Assign NaN or default values
                    for feature in features.values():
                        column_name = f"{time_type}_{abs(p)}hr_{'before' if p < 0 else 'after'}_{feature}" if p != 0 else f"SCHEDULED_{stat_name}_datetime_{feature}"
                        df.at[index, column_name] = None
                    continue

                # Assign weather features to the DataFrame
                for feature_key, feature_name in features.items():
                    if p == 0:
                        column_name = f"SCHEDULED_{stat_name}_DATETIME_{feature_name}"
                    else:
                        relative = 'before' if p < 0 else 'after'
                        column_name = f"{time_type}_{abs(p)}hr_{relative}_{feature_name}"

                    df.at[index, column_name] = temp_weather.iloc[0][feature_name]

    return df



def transform(df):

    column_order = [
        'DISTANCE', 'SCHEDULED_DEPARTURE_DATETIME_tmpc',
        'SCHEDULED_DEPARTURE_DATETIME_sped',
        'SCHEDULED_DEPARTURE_DATETIME_p01m',
        'SCHEDULED_DEPARTURE_DATETIME_vsby', 'departure_6hr_before_tmpc',
        'departure_6hr_before_sped', 'departure_6hr_before_p01m',
        'departure_6hr_before_vsby', 'departure_1hr_before_tmpc',
        'departure_1hr_before_sped', 'departure_1hr_before_p01m',
        'departure_1hr_before_vsby', 'departure_1hr_after_tmpc',
        'departure_1hr_after_sped', 'departure_1hr_after_p01m',
        'departure_1hr_after_vsby', 'departure_6hr_after_tmpc',
        'departure_6hr_after_sped', 'departure_6hr_after_p01m',
        'departure_6hr_after_vsby', 'SCHEDULED_ARRIVAL_DATETIME_tmpc',
        'SCHEDULED_ARRIVAL_DATETIME_sped', 'SCHEDULED_ARRIVAL_DATETIME_p01m',
        'SCHEDULED_ARRIVAL_DATETIME_vsby', 'arrival_6hr_before_tmpc',
        'arrival_6hr_before_sped', 'arrival_6hr_before_p01m',
        'arrival_6hr_before_vsby', 'arrival_1hr_before_tmpc',
        'arrival_1hr_before_sped', 'arrival_1hr_before_p01m',
        'arrival_1hr_before_vsby', 'arrival_1hr_after_tmpc',
        'arrival_1hr_after_sped', 'arrival_1hr_after_p01m',
        'arrival_1hr_after_vsby', 'arrival_6hr_after_tmpc',
        'arrival_6hr_after_sped', 'arrival_6hr_after_p01m',
        'arrival_6hr_after_vsby', 'AVERAGE_WEATHER_DELAY'
    ]

    scaler = load('data/inference/scaler.joblib')

    drop_features = [
        "origin_airport",
        "destination_airport",
        "airline",
        "flight_number",
        "departure_datetime",
        "arrival_datetime",
    ]

    df = df.drop(columns=drop_features, )

    df = df[column_order]
    df_transformed = scaler.transform(df)
    df_transformed = pd.DataFrame(df_transformed, columns=df.columns, index=df.index)

    return df_transformed


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth using the Haversine formula.

    Parameters:
    - lat1, lon1: Latitude and longitude of point 1 in decimal degrees.
    - lat2, lon2: Latitude and longitude of point 2 in decimal degrees.

    Returns:
    - Distance in miles.
    """
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Radius of Earth in miles
    radius_miles = 3958.8
    return radius_miles * c