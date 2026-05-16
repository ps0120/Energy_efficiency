from __future__ import annotations

import pandas as pd
import streamlit as st

import attention_lstm_backend as backend


st.set_page_config(page_title="MMU Energy Prediction", layout="wide")

st.markdown("### MMU Energy Prediction")


@st.cache_data(show_spinner=False)
def _load_history_cached(path: str) -> pd.DataFrame:
	return backend.load_history(path)


def _init_state() -> None:
	if "status" not in st.session_state:
		st.session_state.status = ""
	if "history_path" not in st.session_state:
		st.session_state.history_path = backend.HISTORY_DEFAULT_PATH
	if "data_df" not in st.session_state:
		st.session_state.data_df = pd.DataFrame(
			columns=[
				"TimeStamp",
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
		)
	if "pred_df" not in st.session_state:
		st.session_state.pred_df = pd.DataFrame(
			columns=["TimeStamp", "Actual Usage Peak (kwh)", "Predicted Usage Peak (kwh)"]
		)
	if "metrics" not in st.session_state:
		st.session_state.metrics = {}


def _to_int(name: str, value: str) -> int:
	try:
		return int(str(value).strip())
	except Exception:
		raise ValueError(f"{name} must be an integer")


def _to_float(name: str, value: str) -> float:
	try:
		return float(str(value).strip())
	except Exception:
		raise ValueError(f"{name} must be a number")


_init_state()

# ==================== Top inputs (match screenshot layout) ====================
col1_0, col1_1, col1_2, col1_3 = st.columns([1, 2, 2, 2])
with col1_0:
	st.write("")
	st.write("Date :")
with col1_1:
	day = st.text_input("Day (DD)*", value="0")
with col1_2:
	month = st.text_input("Month (MM)*", value="0")
with col1_3:
	year = st.text_input("Year (YYYY)*", value="0")

col2_1, col2_2, col2_3, col2_4 = st.columns(4)
with col2_1:
	t_lockdown = st.text_input("Type of lockdown *", value="0")
with col2_2:
	temperature = st.text_input("Temperature (°C) *", value="")
with col2_3:
	humidity = st.text_input("Relative Humidity (%) *", value="")
with col2_4:
	pressure = st.text_input("Pressure (hPa) *", value="")

col3_1, col3_2, col3_3, col3_4 = st.columns(4)
with col3_1:
	rainfall_duration = st.text_input("Rainfall Duration (min) *", value="")
with col3_2:
	rainfall_amount = st.text_input("Rainfall amount (mm) *", value="")
with col3_3:
	wind_speed = st.text_input("Wind speed (m/s) *", value="")
with col3_4:
	t_day = st.text_input("Type of day *", value="0")

st.write("")

# ==================== Buttons (Enter / Clear) ====================
b_col1, b_col2, b_col3, b_col4, b_col5 = st.columns(5)
with b_col2:
	btn_enter = st.button("Enter", use_container_width=True)
with b_col3:
	btn_clear = st.button("Clear", use_container_width=True)


def _handle_enter() -> None:
	history = _load_history_cached(st.session_state.history_path)

	day_i = _to_int("Day", day)
	month_i = _to_int("Month", month)
	year_i = _to_int("Year", year)
	tday_i = _to_int("Type of day", t_day)
	tlock_i = _to_int("Type of lockdown", t_lockdown)

	temp_f = _to_float("Temperature", temperature)
	hum_f = _to_float("Relative Humidity", humidity)
	pres_f = _to_float("Pressure", pressure)
	rain_dur_f = _to_float("Rainfall Duration", rainfall_duration)
	rain_amt_f = _to_float("Rainfall amount", rainfall_amount)
	wind_f = _to_float("Wind speed", wind_speed)

	row = backend.build_user_row(
		history,
		day=day_i,
		month=month_i,
		year=year_i,
		type_of_day=tday_i,
		type_of_lockdown=tlock_i,
		temperature=temp_f,
		humidity=hum_f,
		pressure=pres_f,
		rainfall_duration=rain_dur_f,
		rainfall_amount=rain_amt_f,
		wind_speed=wind_f,
	)

	st.session_state.data_df = pd.concat(
		[st.session_state.data_df, pd.DataFrame([row])], ignore_index=True
	)
	st.session_state.status = "Database is succesfully updated"


def _handle_clear() -> None:
	st.session_state.data_df = st.session_state.data_df.iloc[0:0]
	st.session_state.pred_df = st.session_state.pred_df.iloc[0:0]
	st.session_state.metrics = {}
	st.session_state.status = ""


if btn_enter:
	try:
		_handle_enter()
	except Exception as e:
		st.session_state.status = str(e)

if btn_clear:
	_handle_clear()


# This corresponds to the command_text Label in Tkinter
if st.session_state.status:
	st.write(st.session_state.status)
else:
	st.write(" ")

# ==================== Data table ====================
st.caption("Data")
st.dataframe(st.session_state.data_df, use_container_width=True, height=260)

st.write("")

# ==================== Buttons (Train / Update) ====================
t_col1, t_col2, t_col3, t_col4, t_col5 = st.columns(5)
with t_col2:
	btn_train = st.button("Train", use_container_width=True)
with t_col3:
	btn_update = st.button("Update", use_container_width=True)


def _handle_train() -> None:
	history = _load_history_cached(st.session_state.history_path)
	with st.spinner("Training model... this may take a while"):
		result = backend.train_attention_lstm(history=history, user_df=st.session_state.data_df)
	st.session_state.pred_df = result.predicted_df
	st.session_state.metrics = result.metrics
	st.session_state.status = "Training complete"


def _handle_update() -> None:
	# In the original Tkinter app, Update displays predictions after Train.
	if st.session_state.pred_df is None or len(st.session_state.pred_df) == 0:
		st.session_state.status = "Please click Train first."
	else:
		st.session_state.status = "Predicted data updated"


if btn_train:
	try:
		_handle_train()
	except Exception as e:
		st.session_state.status = f"Train failed: {e}"

if btn_update:
	_handle_update()


# ==================== Predicted table ====================
st.caption("Predicted Data")
st.dataframe(st.session_state.pred_df, use_container_width=True, height=260)

# Show metrics without adding extra UI elements (simple text only)
if st.session_state.metrics:
	metrics_text = ", ".join([f"{k}: {v:.4f}" for k, v in st.session_state.metrics.items()])
	st.write(metrics_text)

