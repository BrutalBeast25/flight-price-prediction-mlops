
import os
import json
import requests
import streamlit as st


# ---------------------------------------------------------
# 1. PROJECT CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

ARTIFACT_DIR = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "v1.0"
)

UI_OPTIONS_PATH = os.path.join(
    ARTIFACT_DIR,
    "ui_options.json"
)

NUMERIC_RANGES_PATH = os.path.join(
    ARTIFACT_DIR,
    "numeric_ranges.json"
)

# API_URL = "http://127.0.0.1:8000"
API_URL = os.environ.get(
    "API_URL",
    "http://127.0.0.1:8000"
)


# ---------------------------------------------------------
# 2. STREAMLIT PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="wide"
)


# ---------------------------------------------------------
# 3. LOAD UI CONFIGURATION
# ---------------------------------------------------------

@st.cache_data
def load_ui_configuration():

    with open(UI_OPTIONS_PATH, "r") as file:
        ui_options = json.load(file)

    with open(NUMERIC_RANGES_PATH, "r") as file:
        numeric_ranges = json.load(file)

    return ui_options, numeric_ranges


try:

    ui_options, numeric_ranges = load_ui_configuration()

except Exception as error:

    st.error(
        "Unable to load UI configuration: {}".format(error)
    )

    st.stop()


# ---------------------------------------------------------
# 4. PAGE HEADER
# ---------------------------------------------------------

st.title("✈️ Flight Price Predictor")

st.markdown(
    "### Estimate your flight ticket price using Machine Learning."
)

st.caption(
    "Powered by an optimized XGBoost model | Model Version 1.0"
)

st.divider()


# ---------------------------------------------------------
# 5. API CONNECTION STATUS
# ---------------------------------------------------------

try:

    health_response = requests.get(
        API_URL + "/health",
        timeout=3
    )

    health_response.raise_for_status()

    api_status = health_response.json()

    st.success(
        "Prediction API connected | Model Version {}".format(
            api_status.get("model_version", "Unknown")
        )
    )

except requests.exceptions.RequestException:

    st.error(
        "Prediction API is unavailable. "
        "Please start the FastAPI server on port 8000."
    )


# ---------------------------------------------------------
# 6. FLIGHT INPUT FORM
# ---------------------------------------------------------

st.subheader("Enter Flight Details")

with st.form("flight_prediction_form"):

    # Airline and class

    col1, col2 = st.columns(2)

    with col1:

        airline = st.selectbox(
            "Airline",
            options=ui_options["airline"],
            format_func=lambda value: value.replace("_", " ")
        )

    with col2:

        travel_class = st.selectbox(
            "Travel Class",
            options=ui_options["class"]
        )


    st.markdown("#### Journey Information")

    col3, col4 = st.columns(2)

    with col3:

        source_city = st.selectbox(
            "Source City",
            options=ui_options["source_city"]
        )

    with col4:

        destination_options = [
            city
            for city in ui_options["destination_city"]
            if city != source_city
        ]

        destination_city = st.selectbox(
            "Destination City",
            options=destination_options
        )


    st.markdown("#### Flight Schedule")

    col5, col6 = st.columns(2)

    with col5:

        departure_time = st.selectbox(
            "Departure Time",
            options=ui_options["departure_time"],
            format_func=lambda value: value.replace("_", " ")
        )

    with col6:

        arrival_time = st.selectbox(
            "Arrival Time",
            options=ui_options["arrival_time"],
            format_func=lambda value: value.replace("_", " ")
        )


    st.markdown("#### Travel Information")

    col7, col8, col9 = st.columns(3)

    with col7:

        stops = st.selectbox(
            "Number of Stops",
            options=ui_options["stops"],
            format_func=lambda value: value.replace("_", " ")
        )

    with col8:

        duration_range = numeric_ranges["duration"]

        duration = st.number_input(
            "Flight Duration (Hours)",
            min_value=float(duration_range["min"]),
            max_value=float(duration_range["max"]),
            value=float(duration_range["default"]),
            step=0.5
        )

    with col9:

        days_range = numeric_ranges["days_left"]

        days_left = st.number_input(
            "Days Left Before Departure",
            min_value=int(days_range["min"]),
            max_value=int(days_range["max"]),
            value=int(round(days_range["default"])),
            step=1
        )


    st.divider()

    predict_button = st.form_submit_button(
        "Predict Flight Price",
        type="primary",
        use_container_width=True
    )


# ---------------------------------------------------------
# 7. GENERATE MODEL INPUT
# ---------------------------------------------------------

if predict_button:

    route = "{}_to_{}".format(
        source_city,
        destination_city
    )

    airline_class = "{}_{}".format(
        airline,
        travel_class
    )

    route_class = "{}_{}".format(
        route,
        travel_class
    )

    features = {
        "airline": airline,
        "source_city": source_city,
        "destination_city": destination_city,
        "route": route,
        "departure_time": departure_time,
        "arrival_time": arrival_time,
        "stops": stops,
        "class": travel_class,
        "duration": float(duration),
        "days_left": int(days_left),
        "airline_class": airline_class,
        "route_class": route_class
    }


    # -----------------------------------------------------
    # 8. CALL FASTAPI PREDICTION ENDPOINT
    # -----------------------------------------------------

    st.divider()

    st.subheader("Prediction Result")

    try:

        with st.spinner("Predicting your flight price..."):

            response = requests.post(
                API_URL + "/predict",
                json={
                    "features": features
                },
                timeout=30
            )

        response.raise_for_status()

        result = response.json()

        predicted_price = result["predicted_price"]


        # -------------------------------------------------
        # 9. DISPLAY PREDICTION
        # -------------------------------------------------

        st.success("Prediction generated successfully!")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                label="Estimated Flight Ticket Price",
                value="₹{:,.2f}".format(predicted_price)
            )

        with result_col2:

            st.metric(
                label="Model Version",
                value=result["model_version"]
            )


        # -------------------------------------------------
        # 10. DISPLAY REQUEST DETAILS
        # -------------------------------------------------

        with st.expander("View submitted flight details"):

            st.json(features)


    except requests.exceptions.HTTPError as error:

        try:
            error_details = response.json()

        except ValueError:
            error_details = str(error)

        st.error(
            "Prediction API returned an error: {}".format(
                error_details
            )
        )


    except requests.exceptions.RequestException as error:

        st.error(
            "Unable to connect to prediction API: {}".format(
                error
            )
        )


    except (KeyError, ValueError) as error:

        st.error(
            "Unexpected prediction response: {}".format(
                error
            )
        )


# ---------------------------------------------------------
# 11. FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "Flight Price Prediction | XGBoost v1.0 | "
    "Machine Learning Engineering Project"
)
