import streamlit as st
import pandas as pd
import pydeck as pdk
import datetime
import pickle
from modules.utils import add_weather_forecast, haversine, transform

# -----------------------
# Streamlit Page Configuration
# -----------------------
st.set_page_config(layout="wide")

# -----------------------
# Load Airports Data
# -----------------------
@st.cache_data
def load_airports():
    return pd.read_csv("data/inference/airports.csv")

airports = load_airports()

# -----------------------
# Load Flight Numbers Dictionary
# -----------------------
@st.cache_data
def load_flight_numbers():
    with open("data/inference/flight_numbers.pkl", "rb") as file:
        loaded_dict = pickle.load(file)
    return loaded_dict

flight_numbers = load_flight_numbers()

# Define the ten airports to highlight
ten_airports_codes = ["ATL", "DEN", "DFW", "IAH", "LAS", "LAX", "LGA", "ORD", "PHX", "SFO"]
ten_airports = airports[airports["IATA_CODE"].isin(ten_airports_codes)].dropna(subset=["LATITUDE", "LONGITUDE"])

# -----------------------
# Load Pre-trained Model
# -----------------------
@st.cache_data
def load_model():
    with open("data/inference/final_model1.sav", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# -----------------------
# Define get_matching_flight_numbers Function
# -----------------------
def get_matching_flight_numbers(origin_code, destination_code, airline_code, flight_dict):

    matching_flights = [
        flight_number for flight_number, info in flight_dict.items()
        if (origin_code, destination_code, airline_code) in info['routes']
    ]

    return matching_flights

# -----------------------
# App Description
# -----------------------
st.title("\u2708\ufe0f Flight Delay Predictor Due to Weather \u2601\ufe0f")
st.markdown(
    """
    **Welcome!** This application predicts flight delays caused by weather conditions. Simply provide your flight details,
    and the app will estimate how much delay you might experience. This is especially useful for planning your trips better.\n\n
    *Disclaimer: This is a prediction based on historical data and realtime weather forecasts and assumes that there is a flight delay!*
    """
)

# -----------------------
# User Input Section
# -----------------------
st.subheader("Select Flight Details")

max_allowed_date = datetime.date.today() + datetime.timedelta(weeks=2)
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

with col1:
    # Selectbox for Origin Airport (displaying AIRPORT names)
    origin_airport = st.selectbox(
        "📍 Origin Airport",
        options=ten_airports["AIRPORT"].sort_values().unique(),
        index=0
    )

with col2:
    # Selectbox for Destination Airport (displaying AIRPORT names)
    destination_airport = st.selectbox(
        "📍 Destination Airport",
        options=ten_airports["AIRPORT"].sort_values().unique(),
        index=1
    )

with col3:
    # Date and Time Input for Departure
    departure_date = st.date_input(
        "🗓️🛫 Departure Date",
        value=datetime.date.today(),
        min_value=datetime.date.today(),
        max_value=max_allowed_date
    )
    departure_time = st.time_input("⏰ Departure Time", value=datetime.time(12, 0))

with col4:
    # Date and Time Input for Arrival
    arrival_date = st.date_input(
        "🗓️🛬 Arrival Date",
        value=datetime.date.today(),
        min_value=datetime.date.today(),
        max_value=max_allowed_date
    )
    arrival_time = st.time_input("⏰ Arrival Time", value=datetime.time(15, 0))

# -----------------------
# Airline Selector
# -----------------------
st.subheader("🛩️ Select Airline")

# Dictionary of Airlines with their IATA Codes
airlines = {
    'American Eagle Airlines Inc.': 'MQ',
    'Atlantic Southeast Airlines': 'EV',
    'Delta Air Lines Inc.': 'DL',
    'Frontier Airlines Inc.': 'F9',
    'JetBlue Airways': 'B6',
    'Skywest Airlines Inc.': 'OO',
    'Southwest Airlines Co.': 'WN',
    'Spirit Air Lines': 'NK',
    'US Airways Inc.': 'US',
    'United Air Lines Inc.': 'UA',
    'Virgin America': 'VX'
}

# Selectbox for Airline Selection
selected_airline = st.selectbox(
    "Choose Your Airline",
    options=airlines.keys(),
    index=0
)

# Retrieve the IATA code for the selected airline
airline_code = airlines[selected_airline]

# -----------------------
# Flight Number Selector
# -----------------------
st.subheader("🔢 Select Flight Number")

# Retrieve origin and destination IATA codes
origin_data = ten_airports[ten_airports["AIRPORT"] == origin_airport].iloc[0]
destination_data = ten_airports[ten_airports["AIRPORT"] == destination_airport].iloc[0]

origin_code = origin_data["IATA_CODE"]
destination_code = destination_data["IATA_CODE"]

# Retrieve matching flight numbers based on IATA codes and airline code
matching_flights = get_matching_flight_numbers(origin_code, destination_code, airline_code, flight_numbers)

if matching_flights:
    selected_flight_number = st.selectbox(
        "Choose Your Flight Number",
        options=matching_flights,
        index=0
    )
else:
    st.warning("🚫 No flight numbers found for the selected origin, destination airports, and airline.")
    selected_flight_number = None

# -----------------------
# Map Visualization
# -----------------------
st.subheader("🗺️ Flight Route Visualization")

# Define the initial view state centered on the US
view_state = pdk.ViewState(latitude=39.8283, longitude=-98.5795, zoom=3)

# Define the airport scatterplot layer
airport_layer = pdk.Layer(
    "ScatterplotLayer",
    data=ten_airports,
    get_position='[LONGITUDE, LATITUDE]',
    get_color='[200, 30, 0, 160]',
    get_radius=80000,  # Radius in meters
    pickable=True,
    auto_highlight=True
)

# Define the flight route line layer
line_data = pd.DataFrame({
    "from_lon": [origin_data["LONGITUDE"]],
    "from_lat": [origin_data["LATITUDE"]],
    "to_lon": [destination_data["LONGITUDE"]],
    "to_lat": [destination_data["LATITUDE"]]
})

line_layer = pdk.Layer(
    "LineLayer",
    data=line_data,
    get_source_position='[from_lon, from_lat]',
    get_target_position='[to_lon, to_lat]',
    get_color='[0, 100, 200, 200]',
    get_width=5
)

# Combine layers and create the Deck
deck = pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=view_state,
    layers=[airport_layer, line_layer],
    tooltip={"text": "{AIRPORT}\n{CITY}, {STATE}"}
)

# Render the map
st.pydeck_chart(deck)

# -----------------------
# Prediction Logic
# -----------------------
st.subheader("📈 Delay Prediction")

# Prediction Button
if st.button("🔮 Predict Delay"):
    if selected_flight_number is None:
        st.error("🚫 Please select a valid flight number to proceed with prediction.")
    else:
        # Combine departure and arrival date and time into datetime objects
        departure_datetime = datetime.datetime.combine(departure_date, departure_time)
        arrival_datetime = datetime.datetime.combine(arrival_date, arrival_time)

        # Determine if departure or arrival is on a weekend
        is_departure_weekend = departure_datetime.weekday() >= 5  # 5 = Saturday, 6 = Sunday
        is_arrival_weekend = arrival_datetime.weekday() >= 5
        weekend_feature = is_departure_weekend or is_arrival_weekend

        # Validate that arrival is after departure
        if departure_date > max_allowed_date or arrival_date > max_allowed_date:
            st.error("🚫 Please select departure and arrival dates within two weeks from today.")
        elif arrival_datetime <= departure_datetime:
            st.error("🚫 Arrival datetime must be after departure datetime.")
        else:
            # Feature Engineering
            features = {
                "DISTANCE": haversine(
                    origin_data["LATITUDE"],
                    origin_data["LONGITUDE"],
                    destination_data["LATITUDE"],
                    destination_data["LONGITUDE"]
                ),
                "AVERAGE_WEATHER_DELAY": flight_numbers[selected_flight_number]['AVERAGE_WEATHER_DELAY'],
                "origin_airport": origin_data["IATA_CODE"],
                "destination_airport": destination_data["IATA_CODE"],
                "airline": selected_airline,
                "flight_number": selected_flight_number,
                "departure_datetime": departure_datetime,
                "arrival_datetime": arrival_datetime,
            }

            # Convert features to DataFrame
            input_df = pd.DataFrame([features])

            # Add weather forecast features
            try:
                input_df = add_weather_forecast(input_df)
            except Exception as e:
                st.error(f"An error occurred while adding weather data: {e}")
                st.stop()

            input_df = transform(input_df)

            # -----------------------
            # One-Hot Encoding for Categorical Variables
            # -----------------------
            # 1. Airlines
            for airline in airlines.keys():
                column_name = f"AIRLINE_{airline}"
                input_df[column_name] = 1 if selected_airline == airline else 0

            # 2. Origin Airports
            origin_airports = ten_airports["AIRPORT"].sort_values().unique()
            for origin in origin_airports:
                column_name = f"Origin_AIRPORT_{origin}"
                input_df[column_name] = 1 if origin == origin_airport else 0

            # 3. Destination Airports
            destination_airports = ten_airports["AIRPORT"].sort_values().unique()
            for destination in destination_airports:
                column_name = f"Destination_AIRPORT_{destination}"
                input_df[column_name] = 1 if destination == destination_airport else 0

            input_df['WEEKEND_1'] = weekend_feature

            # -----------------------
            # Ensuring All Expected Columns are Present
            # -----------------------
            # Define expected columns based on model training
            expected_columns = [
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
                'arrival_6hr_after_vsby', 'AVERAGE_WEATHER_DELAY', 'WEEKEND_1',
                'AIRLINE_American Eagle Airlines Inc.',
                'AIRLINE_Atlantic Southeast Airlines', 'AIRLINE_Delta Air Lines Inc.',
                'AIRLINE_Frontier Airlines Inc.', 'AIRLINE_JetBlue Airways',
                'AIRLINE_Skywest Airlines Inc.', 'AIRLINE_Southwest Airlines Co.',
                'AIRLINE_Spirit Air Lines', 'AIRLINE_US Airways Inc.',
                'AIRLINE_United Air Lines Inc.', 'AIRLINE_Virgin America',
                'Origin_AIRPORT_Dallas/Fort Worth International Airport',
                'Origin_AIRPORT_Denver International Airport',
                'Origin_AIRPORT_George Bush Intercontinental Airport',
                'Origin_AIRPORT_Hartsfield-Jackson Atlanta International Airport',
                'Origin_AIRPORT_LaGuardia Airport (Marine Air Terminal)',
                'Origin_AIRPORT_Los Angeles International Airport',
                'Origin_AIRPORT_McCarran International Airport',
                'Origin_AIRPORT_Phoenix Sky Harbor International Airport',
                'Origin_AIRPORT_San Francisco International Airport',
                'Destination_AIRPORT_Dallas/Fort Worth International Airport',
                'Destination_AIRPORT_Denver International Airport',
                'Destination_AIRPORT_George Bush Intercontinental Airport',
                'Destination_AIRPORT_Hartsfield-Jackson Atlanta International Airport',
                'Destination_AIRPORT_LaGuardia Airport (Marine Air Terminal)',
                'Destination_AIRPORT_Los Angeles International Airport',
                'Destination_AIRPORT_McCarran International Airport',
                'Destination_AIRPORT_Phoenix Sky Harbor International Airport',
                'Destination_AIRPORT_San Francisco International Airport'
            ]

            # Reorder columns to match model's expectation
            input_df = input_df[expected_columns]

            # -----------------------
            # Model Prediction
            # -----------------------
            try:
                predicted_delay = model.predict(input_df)[0]
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.stop()

            # Display the prediction result
            st.success(
                f"**Predicted Delay Due to Weather:** {predicted_delay:.1f} minutes 🕒\n\n"
                f"*Note: This is just a prediction. The actual delay may lay between {round(predicted_delay-25, 0)} and {round(predicted_delay+25, 0)}.*"
            )