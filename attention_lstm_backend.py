from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from array import array

from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path


# Keep file/path names identical to try3 - attention-lstm.py
HISTORY_DEFAULT_PATH = "MMU Energy Consumption 2018-2021.xlsx"
DEFAULT_MODEL_PATH = "lstm_model.keras"
DEFAULT_SCALER_PATH = "scaler.pkl"

FEATURE_COLS = [
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

N_DAYS = 7
N_FEATURES = 9
HOLDOUT_FRACTION = 0.2
TEST_TAIL_SIZE = 100
USER_YEAR_MIN = 2021
USER_YEAR_MAX = 2023
HISTORY_MAX_YEAR = 2021


def model_artifacts_exist(
    model_path: str = DEFAULT_MODEL_PATH,
    scaler_path: str = DEFAULT_SCALER_PATH,
) -> bool:
    return Path(model_path).exists() and Path(scaler_path).exists()


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
   
    if day < 1 or day > 31 or month < 1 or month > 12:
        raise ValueError("Invalid date entered")

    if year < USER_YEAR_MIN or year > USER_YEAR_MAX:
        raise ValueError(f"Invalid year, must be between {USER_YEAR_MIN}-{USER_YEAR_MAX}")

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

    filtered = df2[
        (df2["Type of day"] == type_of_day) &
        (df2["Type of Lockdown"] == type_of_lockdown)
    ]

    if len(filtered) > 0:
        energy = filtered["Usage Peak (kwh)"].mean()
    else:
       
        energy = df2["Usage Peak (kwh)"].mean()

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


def _inverse_transform_usage_peak(scaler: MinMaxScaler, scaled_targets: np.ndarray) -> np.ndarray:
    inv_full = np.zeros((len(scaled_targets), N_FEATURES))
    inv_full[:, 0] = scaled_targets
    return scaler.inverse_transform(inv_full)[:, 0]


def _compute_holdout_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {
        " RMSE": rmse,
        " MAE": mae,
        " MAPE(%)": mape,
        " Accuracy(%)": float(max(0, 100 - mape)),
        "RMSE Percentage": float((rmse / np.mean(y_true)) * 100),
        "Median Absolute Error": float(median_absolute_error(y_true, y_pred)),
    }


def _predict_user_rows(
    *,
    history_train: pd.DataFrame,
    user_df: pd.DataFrame,
    model,
    scaler: MinMaxScaler,
) -> pd.DataFrame:
    if user_df is None or len(user_df) == 0:
        return pd.DataFrame(columns=["TimeStamp", "Predicted Usage Peak (kwh)"])

    df = user_df.copy()
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], dayfirst=True)
    df["TimeStamp"] = df["TimeStamp"].dt.strftime("%Y-%m-%d")

    dataset = pd.concat([history_train, df], ignore_index=True)
    values = dataset[FEATURE_COLS].values
    num_user = len(df)

    scaled = scaler.transform(values)
    reframed = series_to_supervised(scaled, N_DAYS, 1)
    reframed_values = reframed.values

    n_history_sequences = len(reframed_values) - num_user
    if n_history_sequences < 0:
        raise ValueError("Not enough history to create sequences for prediction")

    obs = N_DAYS * N_FEATURES
    user_sequences = reframed_values[n_history_sequences:, :]
    user_X = user_sequences[:, :obs].reshape((len(user_sequences), N_DAYS, N_FEATURES))

    yhat = model.predict(user_X, verbose=0)
    inv_yhat = _inverse_transform_usage_peak(scaler, yhat[:, 0])

    start_index = len(dataset) - num_user
    timestamps = dataset["TimeStamp"].iloc[start_index : start_index + len(inv_yhat)].astype(str)

    return pd.DataFrame(
        {
            "TimeStamp": list(timestamps.values),
            "Predicted Usage Peak (kwh)": inv_yhat[: len(timestamps)],
        }
    )


def train_attention_lstm(
    *,
    history: Optional[pd.DataFrame] = None,
    user_df: pd.DataFrame,
    history_path: str = HISTORY_DEFAULT_PATH,
    save_model_path: Optional[str] = None,
    save_scaler_path: Optional[str] = None,
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
            Softmax,
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

    if user_df is None:
        raise ValueError("user_df is required")

    _ensure_history_file_exists(history_path)
    history_train = load_history(history_path)
    history_values = history_train[FEATURE_COLS].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(history_values)
    scaled_history = scaler.transform(history_values)

    reframed_history = series_to_supervised(scaled_history, N_DAYS, 1)
    values_history = reframed_history.values

    if len(values_history) < 5:
        raise ValueError("Not enough historical data to train/evaluate")

    test_size = min(TEST_TAIL_SIZE, len(values_history) - 1)
    if test_size < 1:
        raise ValueError("Not enough historical data to create a test split")

    split_point = len(values_history) - test_size
    train_pool_idx = np.arange(split_point)
    test_idx = np.arange(split_point, len(values_history))

    val_size = max(1, int(round(len(train_pool_idx) * 0.1))) if len(train_pool_idx) > 2 else 0
    if val_size > 0:
        val_idx = train_pool_idx[-val_size:]
        train_idx = train_pool_idx[:-val_size]
    else:
        val_idx = np.array([], dtype=int)
        train_idx = train_pool_idx

    train = values_history[train_idx]
    test = values_history[test_idx]
    val = values_history[val_idx] if len(val_idx) > 0 else None

    obs = N_DAYS * N_FEATURES
    train_X, train_y = train[:, :obs], train[:, -N_FEATURES]
    test_X, test_y = test[:, :obs], test[:, -N_FEATURES]

    if val is not None:
        val_X, val_y = val[:, :obs], val[:, -N_FEATURES]
    else:
        val_X, val_y = None, None

    train_X = train_X.reshape((train_X.shape[0], N_DAYS, N_FEATURES))
    test_X = test_X.reshape((test_X.shape[0], N_DAYS, N_FEATURES))
    if val_X is not None:
        val_X = val_X.reshape((val_X.shape[0], N_DAYS, N_FEATURES))

    inputs = Input(shape=(train_X.shape[1], train_X.shape[2]))
    # Reduced LSTM units to 64 to prevent overfitting on small datasets
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(inputs)

    # Correct Temporal Attention Mechanism
    # Calculate score for each time step (day)
    attention_scores = Dense(1, activation="tanh")(lstm_out)
    # Apply softmax across the temporal dimension (axis=1)
    attention_probs = Softmax(axis=1, name="attention_vec")(attention_scores)
    # Multiply LSTM outputs by temporal weights
    attention_mul = multiply([lstm_out, attention_probs])

    attention_mul_compressed = Flatten()(attention_mul)
    dropout_layer = Dropout(0.5)(attention_mul_compressed)
    output = Dense(1)(dropout_layer)

    model = Model(inputs=inputs, outputs=output)
    model.compile(
        loss="huber",  # In Keras/TensorFlow, the string identifier is "huber", not "huber_loss"
        optimizer=Adam(learning_rate=0.0001),
        metrics=["mae"],
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)

    model.fit(
        train_X,
        train_y,
        epochs=500,
        batch_size=32,
        validation_data=(val_X, val_y) if val_X is not None else None,
        verbose=0,
        shuffle=False,
        callbacks=[early_stop] if val_X is not None else [],
    )

    model.evaluate(test_X, test_y, batch_size=32, verbose=0)
    yhat = model.predict(test_X, verbose=0)

    inv_yhat = _inverse_transform_usage_peak(scaler, np.ravel(yhat))
    inv_y = _inverse_transform_usage_peak(scaler, np.ravel(test_y))
    metrics = _compute_holdout_metrics(inv_y, inv_yhat)

    predicted_df = _predict_user_rows(
        history_train=history_train,
        user_df=user_df,
        model=model,
        scaler=scaler,
    )
    full_data = array("f", [float(x) for x in predicted_df["Predicted Usage Peak (kwh)"].tolist()])

    # Optionally save trained model and fitted scaler for later inference
    if save_model_path and save_scaler_path:
        try:
            Path(save_model_path).parent.mkdir(parents=True, exist_ok=True)
            model.save(save_model_path)
            Path(save_scaler_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, save_scaler_path)
        except Exception:
            # Do not fail the training if saving fails; just continue
            pass

    return TrainResult(predicted_df=predicted_df, metrics=metrics, full_data=full_data)


def load_saved_model_and_scaler(model_path: str, scaler_path: str):
    """Load a saved Keras model and a scaler (joblib).

    Raises FileNotFoundError if files are missing.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")
    if not Path(scaler_path).exists():
        raise FileNotFoundError(f"Saved scaler not found: {scaler_path}")

    try:
        from tensorflow.keras.models import load_model as keras_load_model  # type: ignore
    except Exception as e:
        raise RuntimeError(f"TensorFlow is not available: {e}")

    model = keras_load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_with_saved_model(*, user_df: pd.DataFrame, model_path: str, scaler_path: str, history_path: str = HISTORY_DEFAULT_PATH) -> TrainResult:
    """Run inference on user-entered rows using a saved model and scaler.

    Metrics are not computed here because user rows have no verified ground-truth labels.
    """
    model, scaler = load_saved_model_and_scaler(model_path, scaler_path)

    _ensure_history_file_exists(history_path)
    history_train = pd.read_excel(history_path)

    predicted_df = _predict_user_rows(
        history_train=history_train,
        user_df=user_df,
        model=model,
        scaler=scaler,
    )
    full_data = array("f", [float(x) for x in predicted_df["Predicted Usage Peak (kwh)"].tolist()])

    return TrainResult(predicted_df=predicted_df, metrics={}, full_data=full_data)
