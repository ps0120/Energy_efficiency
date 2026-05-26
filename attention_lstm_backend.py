from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from array import array

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


# Keep file/path names identical to try3 - attention-lstm.py
HISTORY_DEFAULT_PATH = "MMU Energy Consumption 2018-2021.xlsx"


def _ensure_history_file_exists(path: str) -> None:
    if os.path.exists(path):
        return
    raise FileNotFoundError(
        "History Excel file not found: "
        f"{path}. Put '{HISTORY_DEFAULT_PATH}' in the repo root (or update HISTORY_DEFAULT_PATH / pass history_path)."
    )


def fix_date_format(date_str):
    if pd.isna(date_str):
        return pd.NaT

    date_str = str(date_str).split(" ")[0]
    parts = date_str.replace("-", "/").split("/")

    if len(parts) != 3:
        return pd.NaT

    try:
        first = int(float(parts[0]))
        second = int(float(parts[1]))
        year = int(float(parts[2]))

        if year < 100:
            year = year + 2000

        # if first > 12，then DD/MM/YYYY
        # if first <= 12，then MM/DD/YYYY
        if first > 12:
            day = first
            month = second
        else:
            month = first
            day = second

        return pd.Timestamp(year=year, month=month, day=day)
    except Exception:
        return pd.NaT


def load_history(path: str = HISTORY_DEFAULT_PATH) -> pd.DataFrame:
    # Match try3: read as str then apply fix_date_format
    _ensure_history_file_exists(path)
    history = pd.read_excel(path, dtype={"TimeStamp": str})
    history["TimeStamp"] = history["TimeStamp"].apply(fix_date_format)
    return history


def estimate_usage_peak_from_history(
    history: pd.DataFrame,
    *,
    day: int,
    month: int,
    year: int,
    type_of_day: int,
    type_of_lockdown: int,
) -> float:
    # Same logic as save_info() in try3
    if day < 1 or day > 31 or month < 1 or month > 12:
        raise ValueError("Invalid date entered")

    if year < 2018 or year > 2022:
        raise ValueError("Invalid year, must be between 2018-2022")

    query_year = year
    if year == 2022:
        query_year = 2021

    df2 = pd.DataFrame()
    df2["Usage Peak (kwh)"] = history["Usage Peak (kwh)"]
    df2["TimeStamp"] = history["TimeStamp"]
    df2["Type of day"] = history["Type of day"]
    df2["Type of Lockdown"] = history["Type of Lockdown"]
    df2["Month"] = df2["TimeStamp"].dt.month
    df2["Year"] = df2["TimeStamp"].dt.year
    df2["Day"] = df2["TimeStamp"].dt.day

    lockdown_mask = df2["Type of Lockdown"] == type_of_lockdown
    day_type_mask = df2["Type of day"] == type_of_day

    filtered_by_type = df2[lockdown_mask & day_type_mask]

    exact_date_mask = (
        (filtered_by_type["Year"] == query_year)
        & (filtered_by_type["Month"] == month)
        & (filtered_by_type["Day"] == day)
    )
    exact_date_data = filtered_by_type[exact_date_mask]

    if len(exact_date_data) == 1:
        energy = exact_date_data["Usage Peak (kwh)"].mean()
    else:
        month_mask = (filtered_by_type["Year"] == query_year) & (
            filtered_by_type["Month"] == month
        )
        month_data = filtered_by_type[month_mask]

        if len(month_data) > 0:
            energy = month_data["Usage Peak (kwh)"].mean()
        else:
            year_mask = filtered_by_type["Year"] == query_year
            year_data = filtered_by_type[year_mask]

            if len(year_data) > 0:
                energy = year_data["Usage Peak (kwh)"].mean()
            else:
                if len(filtered_by_type) > 0:
                    energy = filtered_by_type["Usage Peak (kwh)"].mean()
                else:
                    energy = 0

    return float(energy)


def build_user_row(
    history: pd.DataFrame,
    *,
    day: int,
    month: int,
    year: int,
    type_of_day: int,
    type_of_lockdown: int,
    temperature: float,
    humidity: float,
    pressure: float,
    rainfall_duration: float,
    rainfall_amount: float,
    wind_speed: float,
) -> Dict[str, object]:
    # Same columns and ordering as `df.loc[...] = [...]` in try3
    energy = estimate_usage_peak_from_history(
        history,
        day=day,
        month=month,
        year=year,
        type_of_day=type_of_day,
        type_of_lockdown=type_of_lockdown,
    )

    important_date = f"{day}/{month}/{year}"

    return {
        "TimeStamp": important_date,
        "Usage Peak (kwh)": energy,
        "Average pressure (Hg)": float(pressure),
        "Average temperature": float(temperature),
        "Average humidity (%)": float(humidity),
        "Average wind speed (m/s)": float(wind_speed),
        "Rainfall duration (min)": float(rainfall_duration),
        "Rainfall amount (mm)": float(rainfall_amount),
        "Type of day": int(type_of_day),
        "Type of Lockdown": int(type_of_lockdown),
    }


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    # Keep behavior identical to try3
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [("var%d(t-%d)" % (j + 1, i)) for j in range(n_vars)]

    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [("var%d(t)" % (j + 1)) for j in range(n_vars)]
        else:
            names += [("var%d(t+%d)" % (j + 1, i)) for j in range(n_vars)]

    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


@dataclass
class TrainResult:
    predicted_df: pd.DataFrame
    metrics: Dict[str, float]
    full_data: array


def train_attention_lstm(
    *,
    history: Optional[pd.DataFrame] = None,
    user_df: pd.DataFrame,
    history_path: str = HISTORY_DEFAULT_PATH,
) -> TrainResult:
    try:
        from tensorflow.keras.callbacks import EarlyStopping  # type: ignore[import-not-found]
        from tensorflow.keras.layers import (  # type: ignore[import-not-found]
            Bidirectional,
            Dense,
            Dropout,
            Flatten,
            Input,
            LSTM,
            multiply,
        )
        from tensorflow.keras.models import Model  # type: ignore[import-not-found]
        from tensorflow.keras.optimizers import Adam  # type: ignore[import-not-found]
    except Exception as e:
        raise RuntimeError(
            "TensorFlow is not available in this environment. "
            "Streamlit Cloud is currently using Python 3.14.x, where TensorFlow wheels are not published. "
            "To enable training, run locally with Python 3.12/3.13, or deploy on a platform where you can choose Python <= 3.13. "
            f"Original import error: {e}"
        )

    # Match try3 train(): it always reloads history from Excel
    if user_df is None:
        raise ValueError("user_df is required")

    df = user_df.copy()

    _ensure_history_file_exists(history_path)
    history_train = pd.read_excel(history_path)

    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], dayfirst=True)
    df["TimeStamp"] = df["TimeStamp"].dt.strftime("%Y-%m-%d")

    dataset = pd.concat([history_train, df], ignore_index=True)

    values = dataset[
        [
            "Usage Peak (kwh)",
            "Average pressure (Hg)",
            "Average temperature",
            "Average humidity (%)",
            "Average wind speed (m/s)",
            "Rainfall duration (min)",
            "Rainfall amount (mm)",
            "Type of day",
            "Type of Lockdown",
        ]
    ].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    n_days = 7
    n_features = 9
    reframed = series_to_supervised(scaled, n_days, 1)
    values = reframed.values

    num_test = len(df) if len(df) > 0 else 10
    n_train_time = len(values) - num_test
    train = values[:n_train_time, :]
    test = values[n_train_time:, :]

    obs = n_days * n_features
    train_X, train_y = train[:, :obs], train[:, -n_features]
    test_X, test_y = test[:, :obs], test[:, -n_features]

    train_X = train_X.reshape((train_X.shape[0], n_days, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_days, n_features))

    inputs = Input(shape=(train_X.shape[1], train_X.shape[2]))
    lstm_out = Bidirectional(LSTM(100, return_sequences=True))(inputs)

    attention_probs = Dense(200, activation="softmax", name="attention_vec")(lstm_out)
    attention_mul = multiply([lstm_out, attention_probs])

    attention_mul_compressed = Flatten()(attention_mul)
    dropout_layer = Dropout(0.5)(attention_mul_compressed)
    output = Dense(1)(dropout_layer)

    model = Model(inputs=inputs, outputs=output)
    model.compile(
        loss="mean_squared_error",
        optimizer=Adam(learning_rate=0.0001),
        metrics=["mae"],
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)

    model.fit(
        train_X,
        train_y,
        epochs=500,
        batch_size=32,
        validation_data=(test_X, test_y),
        verbose=0,
        shuffle=False,
        callbacks=[early_stop],
    )

    model.evaluate(test_X, test_y, batch_size=32, verbose=0)
    yhat = model.predict(test_X, verbose=0)

    inv_yhat_full = np.zeros((len(yhat), n_features))
    inv_yhat_full[:, 0] = yhat[:, 0]
    inv_yhat = scaler.inverse_transform(inv_yhat_full)[:, 0]

    inv_y_full = np.zeros((len(test_y), n_features))
    inv_y_full[:, 0] = test_y
    inv_y = scaler.inverse_transform(inv_y_full)[:, 0]

    inv_yhat_median = np.median(inv_yhat)
    inv_yhat_std = np.std(inv_yhat)

    for i in range(len(inv_yhat)):
        if abs(inv_yhat[i] - inv_yhat_median) > 2 * inv_yhat_std:
            if i == 0 and len(inv_yhat) > 1:
                inv_yhat[i] = inv_yhat[i + 1]
            elif i > 0:
                inv_yhat[i] = (inv_yhat[i - 1] + inv_yhat[i]) / 2

    rmse = float(np.sqrt(mean_squared_error(inv_y, inv_yhat)))
    mae = float(mean_absolute_error(inv_y, inv_yhat))
    mape = float(np.mean(np.abs((inv_y - inv_yhat) / inv_y)) * 100)
    accuracy = float(max(0, 100 - mape))
    rmse_percentage = float((rmse / np.mean(inv_y)) * 100)

    from sklearn.metrics import median_absolute_error

    mdae = float(median_absolute_error(inv_y, inv_yhat))

    arr_actual = array("f", [])
    arr_forecast = array("f", [])

    for i in range(len(inv_y)):
        number = len(dataset) - num_test + i
        actual = dataset.iloc[number]["Usage Peak (kwh)"]
        forecast = inv_yhat[i]
        arr_actual.append(float(actual))
        arr_forecast.append(float(forecast))

    full_data = arr_forecast

    start_index = len(dataset) - num_test
    timestamps = dataset["TimeStamp"].iloc[start_index : start_index + len(inv_y)].astype(str)

    predicted_df = pd.DataFrame(
        {
            "TimeStamp": list(timestamps.values),
            "Actual Usage Peak (kwh)": inv_y[: len(timestamps)],
            "Predicted Usage Peak (kwh)": inv_yhat[: len(timestamps)],
        }
    )

    metrics = {
        "Test RMSE": rmse,
        "Test MAE": mae,
        "Test MAPE(%)": mape,
        "Test Accuracy(%)": accuracy,
        "RMSE Percentage": rmse_percentage,
        "Median Absolute Error": mdae,
    }

    return TrainResult(predicted_df=predicted_df, metrics=metrics, full_data=full_data)
