import streamlit as st
import requests
import datetime
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

API_URL = "http://127.0.0.1:8000/predict"
API_URL_GET = "http://127.0.0.1:8000/predict_get"
API_URL_POST = "http://127.0.0.1:8000/predict_post/"
API_URL_POST_SIMU = "http://127.0.0.1:8000/predict_post_simu/"

# left panel management
with st.sidebar:
    st.header('DEFINE YOUR SCENARIO', divider='gray')

    with st.expander("Electricity Price YoY increase :thermometer:"):
        elec_price_yoy =st.selectbox(key='elec_yoy',
                                    label = 'Intensity',
                                    index=6,
                                    options= ('5%','4%','3%','2%','baseline','0%',
                                              '-1%','-2%','-3%','-4%','-5%')
                                    )

    with st.expander("Temperature :thermometer:"):
        temp_scenario = st.selectbox(key='temp_scenario',
                                    label = 'Scenario',
                                    options= ('SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5'))

    with st.expander("European conflict :gun:"):
        war_EU_start = st.date_input(key='war_EU_start',
                                    label='Start',
                                    min_value=datetime(2024, 1, 1, 00, 00, 00),
                                    max_value=datetime(2030, 12, 31, 00, 00, 00),
                                    value=datetime.now())
        war_EU_end = st.date_input(key='war_EU_end',
                                    label='End',
                                    min_value=datetime(2024, 1, 1, 00, 00, 00),
                                    max_value=datetime(2030, 12, 31, 00, 00, 00),
                                    value=datetime.now())
        war_EU_intensity = st.selectbox(key='war_EU_intensity',
                                    label = 'Intensity',
                                    options= ('NONE', 'SMALL', 'MEDIUM', 'LARGE', 'HIGH'))

    with st.expander("Middle-East conflict :gun:"):
        war_ME_start = st.date_input(key='war_ME_start',
                                    label='Start',
                                    min_value=datetime(2024, 1, 1, 00, 00, 00),
                                    max_value=datetime(2030, 12, 31, 00, 00, 00),
                                    value=datetime.now())
        war_ME_end = st.date_input(key='war_ME_end',
                                    label='End',
                                    min_value=datetime(2024, 1, 1, 00, 00, 00),
                                    max_value=datetime(2030, 12, 31, 00, 00, 00),
                                    value=datetime.now())
        war_ME_intensity = st.selectbox(key='war_ME_intensity',
                                    label = 'Intensity',
                                    options= ('NONE', 'SMALL', 'MEDIUM', 'LARGE', 'HIGH'))

    with st.expander("Pandemic :skull_and_crossbones:"):
        covid_start = st.date_input(key='covid_start',
                                    label='Start',
                                    min_value=datetime(2024, 1, 1, 00, 00, 00),
                                    max_value=datetime(2030, 12, 31, 00, 00, 00),
                                    value=datetime.now())
        covid_end = st.date_input(key='covid_end',
                                    label='End',
                                    min_value=datetime(2024, 1, 1, 00, 00, 00),
                                    max_value=datetime(2030, 12, 31, 00, 00, 00),
                                    value=datetime.now())
        covid_intensity = st.selectbox(key='covid_intensity',
                                    label = 'Intensity',
                                    options= ('NONE', 'SMALL', 'MEDIUM', 'LARGE', 'HIGH'))


    with st.expander("Lockdown :biohazard_sign:"):
        lockdown_start = st.date_input(key='lockdown_start',
                                        label='Start',
                                        min_value=datetime(2024, 1, 1, 00, 00, 00),
                                        max_value=datetime(2030, 12, 31, 00, 00, 00),
                                        value=datetime.now())
        lockdown_end = st.date_input(key='lockdown_end',
                                    label='End',
                                    min_value=datetime(2024, 1, 1, 00, 00, 00),
                                    max_value=datetime(2030, 12, 31, 00, 00, 00),
                                    value=datetime.now())

# capture event_params
    event_params = {
        'covid':{
            'start':covid_start,
            'end': covid_end,
            'intensity': covid_intensity
            },

        'lockdown':{
            'start': lockdown_start,
            'end': lockdown_end
            },

        'war_EU': {
            'start': war_EU_start,
            'end': war_EU_end,
            'intensity': war_EU_intensity
            },

        'war_ME': {
            'start':war_ME_start,
            'end': war_ME_end,
            'intensity': war_ME_intensity
            },

        'temp': {
            'scenario': temp_scenario
            },

        'elec_price': {
            'yoy_increase': elec_price_yoy
            }
        }


    predict_go = st.button("Predict")

if predict_go == True:
    # capture event_params
    event_params = {
        'covid':{
            'start':str(covid_start)+" 00:00:00",
            'end': str(covid_end)+" 00:00:00",
            'intensity': covid_intensity
            },

        'lockdown':{
            'start': str(lockdown_start)+" 00:00:00",
            'end': str(lockdown_end)+" 00:00:00"
            },

        'war_EU': {
            'start': str(war_EU_start)+" 00:00:00",
            'end': str(war_EU_end)+" 00:00:00",
            'intensity': war_EU_intensity
            },

        'war_ME': {
            'start':str(war_ME_start)+" 00:00:00",
            'end': str(war_ME_end)+" 00:00:00",
            'intensity': war_ME_intensity
            },

        'temp': {
            'scenario': temp_scenario
            },

        'elec_price': {
            'yoy_increase': elec_price_yoy
            }
        }



# # POST INITIAL
# with st.container():
#     st.header("Adjusted scenario")


#     if predict_go == True:
#         response = requests.post(API_URL_POST, json = event_params)
#         if response.status_code == 200:
#             predictions_data = response.json()
#             elect_pred = predictions_data.get("electricity_demand", [])
#             temp_pred = predictions_data.get("temperature", [])
#             covid_pred = predictions_data.get("pandemic", [])
#             #st.write(predictions_data)

#             start_date = datetime(2013, 1, 1, 0, 0)

#             timestamps = [start_date + timedelta(hours=i) for i in range(len(elect_pred))]

#             assert len(elect_pred) == len(timestamps), "Lengths of predictions and timestamps do not match."

#             df_predictions = pd.DataFrame({
#                 'timestamp': timestamps,
#                 'demand': elect_pred,
#                 'temperature': temp_pred,
#                 'pandemic': covid_pred
#             })

#             fig1 = px.line(df_predictions, x='timestamp', y='temperature', title="Predicted temperature 2024")
#             fig2 = px.line(df_predictions, x='timestamp', y='demand', title="Predicted Electricity Demand for 2024")
#             fig3 = px.line(df_predictions, x='timestamp', y='pandemic', title="Pandemic intensity for 2024")
#             st.plotly_chart(fig1)
#             st.plotly_chart(fig2)
#             st.plotly_chart(fig3)
#         else:
#             st.error(f"Failed to get prediction from the API: {response.reason}")

st.divider()

# POST SIMU !!!!
with st.container():
    st.header("Adjusted scenario")


    if predict_go == True:
        response = requests.post(API_URL_POST_SIMU, json = event_params)
        if response.status_code == 200:
            predictions_data = response.json()
            elect_pred = predictions_data.get("electricity_demand", [])
            temp_pred = predictions_data.get("temperature", [])
            covid_pred = predictions_data.get("pandemic", [])
            nd_pred = predictions_data.get("nd_pred", [])
            #st.write(predictions_data)

            start_date = datetime(2013, 1, 1, 0, 0)

            timestamps = [start_date + timedelta(hours=i) for i in range(len(elect_pred))]

            assert len(elect_pred) == len(timestamps), "Lengths of predictions and timestamps do not match."

            df_predictions = pd.DataFrame({
                'timestamp': timestamps,
                'demand': elect_pred,
                'temperature': temp_pred,
                'pandemic': covid_pred,
                'nd_pred': nd_pred
            })

            df_nd = df_predictions[96432:105000]
            # start_date_nd = datetime(2024, 1, 1, 0, 0)
            # timestamps_nd = [start_date_nd + timedelta(hours=i) for i in range(df_nd)]

            #df_nd['timestamp_nd'] = [start_date_nd + timedelta(hours=i) for i in range(df_nd)]



            fig1 = px.line(df_predictions, x='timestamp', y='temperature', title="Predicted temperature 2024")
            fig2 = px.line(df_predictions, x='timestamp', y='demand', title="Predicted Electricity Demand for 2024")
            fig3 = px.line(df_predictions, x='timestamp', y='pandemic', title="Pandemic intensity")
            fig4 = px.line(df_nd, x='timestamp', y='nd_pred', title="Forecasted demand")
            #fig4 = px.line(df_nd, x='timestamp_nd', y='nd_pred', title="Forecasted demand")

            st.plotly_chart(fig4)
            st.plotly_chart(fig1)
            st.plotly_chart(fig3)

        else:
            st.error(f"Failed to get prediction from the API: {response.reason}")

st.divider()

# with st.container():
#     st.header(":flag-gb: Electricity Demand Forecast (2024)")


#     if predict_go == True:
#         response = requests.get(API_URL)
#         if response.status_code == 200:
#             predictions_data = response.json()
#             predictions = predictions_data.get("electricity_demand", [])

#             start_date = datetime(2024, 1, 1, 0, 0)

#             timestamps = [start_date + timedelta(hours=i) for i in range(len(predictions))]

#             assert len(predictions) == len(timestamps), "Lengths of predictions and timestamps do not match."

#             df_predictions = pd.DataFrame({
#                 'timestamp': timestamps,
#                 'demand': predictions
#             })

#             fig = px.line(df_predictions, x='timestamp', y='demand', title="Predicted Electricity Demand for 2024")
#             st.plotly_chart(fig)
#         else:
#             st.error(f"Failed to get prediction from the API: {response.reason}")

# st.divider()













# with st.container():
#     st.header("Metrics")

#     st.metric(label="Temperature", value="70 °F", delta="1.2 °F")


# @st.experimental_memo
# def load_real_data():
#     real_data_path = 'raw_data/2023_noNA_clean.csv'
#     df_real = pd.read_csv(real_data_path)
#     df_real['settlement_date'] = pd.to_datetime(df_real['settlement_date'])
#     # Filter to include only data from 2023
#     df_real = df_real[df_real['settlement_date'].dt.year == 2023]
#     return df_real.rename(columns={'nd': 'demand'})  # Rename 'nd' to 'demand' for consistency

# # Function to fetch predicted data
# @st.experimental_memo
# def fetch_predictions():
#     API_URL = "http://127.0.0.1:8000/predict"
#     response = requests.get(API_URL)
#     if response.status_code == 200:
#         return response.json()['electricity_demand']
#     else:
#         st.error(f"Failed to get prediction from the API: {response.reason}")
#         return []

# st.title("Electricity Demand: Real vs Predicted for 2023")

# # Add a button to trigger prediction fetching
# if st.button("Predict"):
#     # Fetch predicted data
#     predicted_data = fetch_predictions()

#     # Load real data
#     df_real = load_real_data()

#     if predicted_data:
#         # Ensure predicted_data length matches df_real, adjust as needed
#         predicted_length = min(len(predicted_data), len(df_real))

#         df_predicted = pd.DataFrame({
#             'settlement_date': df_real['settlement_date'].iloc[:predicted_length],
#             'demand': predicted_data[:predicted_length]
#         })

#         # Plotting both real and predicted data on the same graph
#         fig = go.Figure()

#         # Add real data trace
#         fig.add_trace(go.Scatter(x=df_real['settlement_date'], y=df_real['demand'],
#                                  mode='lines', name='Real Data',
#                                  line=dict(color='blue')))

#         # Add predicted data trace
#         fig.add_trace(go.Scatter(x=df_predicted['settlement_date'], y=df_predicted['demand'],
#                                  mode='lines', name='Predicted Data',
#                                  line=dict(color='red')))

#         fig.update_layout(title='Electricity Demand: Real vs Predicted for 2023',
#                           xaxis_title='Date',
#                           yaxis_title='Electricity Demand',
#                           legend_title='Source')

#         st.plotly_chart(fig)
