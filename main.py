from ml_logic.data import clean_data, transform_date
from ml_logic.model import get_baseline, preprocess_data_GRU, initialize_GRU, compile_GRU, train_GRU, evaluate_model
from ml_logic.preprocessor import preprocess_features
from ml_logic.registry import load_custom_model, save_model, save_results
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
from pathlib import Path
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta

def run_baseline():
    data = pd.read_csv('raw_data/historic_demand_2009_2024.csv')
    data_clean = clean_data(data)
    print(data_clean.columns)
    date_transformed = transform_date(data_clean)
    print(date_transformed.columns)
    result = pd.DataFrame(get_baseline(date_transformed))

    return result

def run_GRU():
    data = pd.read_csv('raw_data/2023_noNA_clean.csv')
    X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data_GRU(data)
    print(X_train_scaled.shape)
    model = initialize_GRU(X_train_scaled, X_test_scaled, y_train, y_test)
    model_compiled = compile_GRU(model)
    trained_model, history = train_GRU(model_compiled, X_test_scaled, y_test)
    eval_metrics = evaluate_model(trained_model, X_test_scaled, y_test)

    try:
        val_mae = np.min(history.history['val_mae'])
    except KeyError:
        print("Key 'val_mae' not found in history. Available keys:", history.history.keys())
        val_mae = None

    params = {
        "context": "train",
        # Add more relevant training parameters here as needed
    }
    metrics = {
        "mae": val_mae,
        # Include additional metrics as needed
    }

    save_model(trained_model, model_name="electricity_demand_gru")
    save_results(params=params, metrics=metrics)

    print("👽:", X_test_scaled.shape)

    np.save('raw_data/X_test_scaled.npy', X_test_scaled)

    return trained_model, X_test_scaled, eval_metrics

def pred_GRU(model, X_test_scaled):
    predictions = model.predict(X_test_scaled)
    print("pred_GRU", predictions)
    return predictions


def run_LSTM():
    data = pd.read_csv('raw_data/2023_noNA_clean.csv')
    data.drop(columns="Unnamed: 0", inplace=True)
    data['settlement_date'] = pd.to_datetime(data['settlement_date'])
    data.set_index('settlement_date', inplace=True)

    columns_to_keep = [
        'nd', 'is_holiday', 'core_CPI', 'mean_wind_lon', 'mean_wind_north',
        'daily_exchange_rate', 'ukraine_intensity', 'monthly_gdp', 'israel_intensity',
        'ww_intensity', '%_elec_vs_total', 'Electricity CPI Index', 'me_intensity',
        'eu_intensity', 'uk_pop', 'daily_death_count', 'north_temp', 'london_temp'
    ]
    data = data[columns_to_keep]

    X_train = data.loc['2018-01-01':'2022-12-31']
    X_test = data.loc['2019-01-01':'2023-12-31']

    select_not_scaler = ['nd', 'is_holiday']
    select_scaler = [col for col in X_test.columns if col not in select_not_scaler]

    scaler = StandardScaler()

    scaler.fit(X_train[select_scaler])
    X_test[select_scaler] = scaler.transform(X_test[select_scaler])


    X_test_scaled = tf.expand_dims(X_test, axis=0)
    X_test_adjusted = X_test_scaled[:, :43800, :]

    print('👺:', X_test_adjusted.shape)

    model_path = 'LSTM_weights/LSTM_12th_Model_EX_5Y_100324'
    model = load_model(model_path)
    print(f"✅ LSTM loaded")

    predictions = model.predict(X_test_adjusted)

    return predictions


def stack():
    X_test_scaled = np.load('raw_data/X_test_scaled.npy')

    gru_model = load_custom_model(model_name="electricity_demand_gru")
    gru_predictions = pred_GRU(gru_model, X_test_scaled)
    print("GRU predictions shape:", gru_predictions.shape)

    lstm_predictions = run_LSTM()
    print("LSTM predictions shape:", lstm_predictions.shape)

    try:
        gru_predictions = np.squeeze(gru_predictions)
        lstm_predictions = np.squeeze(lstm_predictions)
        if gru_predictions.shape != lstm_predictions.shape:
            raise ValueError("GRU and LSTM predictions have incompatible shapes after squeeze.")
    except ValueError as e:
        print(e)
        raise

    stacked_predictions = np.mean([gru_predictions, lstm_predictions], axis=0)
    print('🥶🥶 Stacked GRU/LSTM preds:', stacked_predictions)

    return stacked_predictions

def run_LSTM_real(df, start, end):
    columns_to_keep = [
        'nd', 'is_holiday', 'core_CPI', 'mean_wind_lon', 'mean_wind_north',
        'daily_exchange_rate', 'ukraine_intensity', 'monthly_gdp', 'israel_intensity',
        'ww_intensity', '%_elec_vs_total', 'Electricity CPI Index', 'me_intensity',
        'eu_intensity', 'uk_pop', 'daily_death_count', 'north_temp', 'london_temp'
    ]
    print(df)
    data = df[columns_to_keep]
    print("data_formated", data.shape)

    #data.index = pd.to_datetime(data.index)

    print("start=", start)
    start_plus_one_year = "2019-01-01 00:00:00"
    #start_plus_one_year = start + pd.DateOffset(years= 1)
    print(start_plus_one_year)

    X_pred = data.loc[start_plus_one_year:end]

    print('X_pred_shape for lstm:', X_pred)

    select_not_scaler = ['nd', 'is_holiday']
    select_scaler = [col for col in X_pred.columns if col not in select_not_scaler]

    scaler = StandardScaler()
    X_pred[select_scaler] = scaler.fit_transform(X_pred[select_scaler])

    X_pred_scaled = tf.expand_dims(X_pred, axis=0)

    print('👺:', X_pred_scaled.shape)

    model_path = 'LSTM_weights/LSTM_12th_Model_EX_5Y_100324'
    model = load_model(model_path)
    print(f"✅ LSTM loaded")

    predictions = model.predict(X_pred_scaled)

    return predictions

def stacked(df, start: str, end: str):

    df.loc['2023-03-26 23:00:00', :] = df.loc['2023-03-26 22:00:00', :].values

    X_pred = df[start:end]
    print(X_pred)

    X_pred = X_pred.drop(columns=["nd", "settlement_date"], axis=1)
    print(X_pred)

    # data_filtered = data['2017-01-01':'2022-12-31']
    # X_train = data_filtered.drop("nd", axis=1)
    # y_train = data_filtered[["nd"]]
    # X_test = data['2023-01-01':'2023-12-31'].drop("nd", axis=1)
    # y_test = data['2023-01-01':'2023-12-31'][['nd']]

    # scaler = MinMaxScaler(feature_range=(0, 1))
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    # X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    # X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))


    scaler = MinMaxScaler(feature_range=(0, 1))
    X_pred_scaled = scaler.fit_transform(X_pred)
    print("X_pred_scaled", X_pred_scaled)
    X_pred_scaled = X_pred_scaled.reshape((X_pred_scaled.shape[0], 1, X_pred_scaled.shape[1]))

    print('X_pred_scaled:', X_pred_scaled.shape)

    gru_model = load_custom_model(model_name="electricity_demand_gru")

    print("GRU model loaded")

    X_pred_scaled_GRU = X_pred_scaled[-8760:]
    print("X_pred_scaled_GRU shape", X_pred_scaled_GRU)
    gru_predictions = pred_GRU(gru_model, X_pred_scaled_GRU)
    print("toto 0")
    print("GRU predictions shape:", gru_predictions.shape)

    print("toto 1")
    print(gru_predictions)
    print("toto 2")

    print(df)
    lstm_predictions = run_LSTM_real(df, start, end)
    print("LSTM predictions shape:", lstm_predictions.shape)
    print("GRU predictions shape:", gru_predictions.shape)

    try:
        gru_predictions = np.squeeze(gru_predictions)
        lstm_predictions = np.squeeze(lstm_predictions)
        if gru_predictions.shape != lstm_predictions.shape:
            raise ValueError("GRU and LSTM predictions have incompatible shapes after squeeze.")
    except ValueError as e:
        print(e)
        raise

    stacked_predictions = np.mean([gru_predictions, lstm_predictions], axis=0)
    print('🥶🥶 Stacked GRU/LSTM preds:', stacked_predictions)

    return stacked_predictions



if __name__ == "__main__":
    # trained_model, X_test_scaled, eval_metrics = run_GRU()
    # predictions = pred(trained_model, X_test_scaled)
    predictions = stack()
    # predictions = run_GRU()

    print(predictions)
    # print(eval_metrics)








if __name__ == "__main__":
    # trained_model, X_test_scaled, eval_metrics = run_GRU()
    # predictions = pred(trained_model, X_test_scaled)
    #predictions = stack()
    predictions = run_GRU()

    print(predictions)
    # print(eval_metrics)
