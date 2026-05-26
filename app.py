from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import attention_lstm_backend as backend


st.set_page_config(page_title="MMU Energy Prediction", layout="wide")

st.markdown("### MMU Energy Prediction")




_RANGES = {
	"day": (1, 31),
	"month": (1, 12),
	"year": (2019, 2022),
	"temperature": (22.0, 31.0),
	"humidity": (58.0, 95.0),
	"pressure": (1006.0, 1014.0),
	"rainfall_duration": (0.0, 60.0),
	"rainfall_amount": (-14.0, 4.0),
	"wind_speed": (0.0, 5.3),
}


_STATE_DIR = Path(".streamlit_state")
_DATA_CSV = _STATE_DIR / "data_df.csv"
_PRED_CSV = _STATE_DIR / "pred_df.csv"
_METRICS_JSON = _STATE_DIR / "metrics.json"


def _restore_state_from_disk() -> None:
	try:
		if _DATA_CSV.exists():
			st.session_state.data_df = pd.read_csv(_DATA_CSV)
		if _PRED_CSV.exists():
			st.session_state.pred_df = pd.read_csv(_PRED_CSV)
		if _METRICS_JSON.exists():
			with _METRICS_JSON.open("r", encoding="utf-8") as f:
				metrics = json.load(f)
				if isinstance(metrics, dict):
					st.session_state.metrics = metrics
	except Exception:
		# Best-effort restore; ignore corruption/IO errors.
		return


def _persist_state_to_disk() -> None:
	try:
		_STATE_DIR.mkdir(parents=True, exist_ok=True)
		st.session_state.data_df.to_csv(_DATA_CSV, index=False)
		st.session_state.pred_df.to_csv(_PRED_CSV, index=False)
		with _METRICS_JSON.open("w", encoding="utf-8") as f:
			json.dump(st.session_state.metrics, f, ensure_ascii=False)
	except Exception:
		# Best-effort persistence; ignore IO errors.
		return


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
	if "_restored_from_disk" not in st.session_state:
		st.session_state._restored_from_disk = True
		_restore_state_from_disk()


def _require_int_range(name: str, value: int, min_value: int, max_value: int) -> None:
	if value < min_value or value > max_value:
		raise ValueError(f"Invalid value: {name} must be between {min_value} and {max_value}")


def _require_float_range(name: str, value: float, min_value: float, max_value: float) -> None:
	if value < min_value or value > max_value:
		raise ValueError(f"Invalid value: {name} must be between {min_value} and {max_value}")


def _validate_enter_inputs() -> str | None:
	try:
		day_i = int(day)
		month_i = int(month)
		year_i = int(year)
		tday_i = int(t_day)
		tlock_i = int(t_lockdown)

		_require_int_range("Day", day_i, int(_RANGES["day"][0]), int(_RANGES["day"][1]))
		_require_int_range(
			"Month", month_i, int(_RANGES["month"][0]), int(_RANGES["month"][1])
		)
		_require_int_range("Year", year_i, int(_RANGES["year"][0]), int(_RANGES["year"][1]))

		if tday_i not in (0, 1):
			raise ValueError(
				"Invalid value: Type of day must be 0 (weekday) or 1 (weekend/holiday)"
			)
		if tlock_i not in (0, 1, 2):
			raise ValueError(
				"Invalid value: Type of lockdown must be 0 (no MCO), 1 (MCO) or 2 (RMCO)"
			)

		temp_f = float(temperature)
		hum_f = float(humidity)
		pres_f = float(pressure)
		rain_dur_f = float(rainfall_duration)
		rain_amt_f = float(rainfall_amount)
		wind_f = float(wind_speed)

		_require_float_range(
			"Temperature (°C)", temp_f, _RANGES["temperature"][0], _RANGES["temperature"][1]
		)
		_require_float_range(
			"Relative Humidity (%)", hum_f, _RANGES["humidity"][0], _RANGES["humidity"][1]
		)
		_require_float_range(
			"Pressure (hPa)", pres_f, _RANGES["pressure"][0], _RANGES["pressure"][1]
		)
		_require_float_range(
			"Rainfall Duration (min)",
			rain_dur_f,
			_RANGES["rainfall_duration"][0],
			_RANGES["rainfall_duration"][1],
		)
		_require_float_range(
			"Rainfall amount (mm)",
			rain_amt_f,
			_RANGES["rainfall_amount"][0],
			_RANGES["rainfall_amount"][1],
		)
		_require_float_range(
			"Wind speed (m/s)",
			wind_f,
			_RANGES["wind_speed"][0],
			_RANGES["wind_speed"][1],
		)
		return None
	except Exception as e:
		return str(e)


def _field_validity() -> dict[str, bool]:
	validity: dict[str, bool] = {}

	def _check(name: str, fn) -> None:
		try:
			fn()
			validity[name] = True
		except Exception:
			validity[name] = False

	_check(
		"Day (DD) [1-31] *",
		lambda: _require_int_range(
			"Day", int(day), int(_RANGES["day"][0]), int(_RANGES["day"][1])
		),
	)
	_check(
		"Month (MM) [1-12] *",
		lambda: _require_int_range(
			"Month", int(month), int(_RANGES["month"][0]), int(_RANGES["month"][1])
		),
	)
	_check(
		"Year (YYYY) [2019-2022] *",
		lambda: _require_int_range(
			"Year", int(year), int(_RANGES["year"][0]), int(_RANGES["year"][1])
		),
	)

	_check(
		"Temperature (°C) [22-31] *",
		lambda: _require_float_range(
			"Temperature (°C)",
			float(temperature),
			_RANGES["temperature"][0],
			_RANGES["temperature"][1],
		),
	)
	_check(
		"Relative Humidity (%) [58-95] *",
		lambda: _require_float_range(
			"Relative Humidity (%)",
			float(humidity),
			_RANGES["humidity"][0],
			_RANGES["humidity"][1],
		),
	)
	_check(
		"Pressure (hPa) [1006-1014] *",
		lambda: _require_float_range(
			"Pressure (hPa)",
			float(pressure),
			_RANGES["pressure"][0],
			_RANGES["pressure"][1],
		),
	)
	_check(
		"Rainfall Duration (min) [0-60] *",
		lambda: _require_float_range(
			"Rainfall Duration (min)",
			float(rainfall_duration),
			_RANGES["rainfall_duration"][0],
			_RANGES["rainfall_duration"][1],
		),
	)
	_check(
		"Rainfall amount (mm) [-14-4] *",
		lambda: _require_float_range(
			"Rainfall amount (mm)",
			float(rainfall_amount),
			_RANGES["rainfall_amount"][0],
			_RANGES["rainfall_amount"][1],
		),
	)
	_check(
		"Wind speed (m/s) [0-5.3] *",
		lambda: _require_float_range(
			"Wind speed (m/s)",
			float(wind_speed),
			_RANGES["wind_speed"][0],
			_RANGES["wind_speed"][1],
		),
	)

	return validity


_init_state()

# ==================== Top inputs (match screenshot layout) ====================
col1_0, col1_1, col1_2, col1_3 = st.columns([1, 2, 2, 2])
with col1_0:
	st.write("")
	st.write("Date :")
with col1_1:
	day = st.text_input(
		"Day (DD) [1-31] *",
		value=str(int(_RANGES["day"][0])),
	)
with col1_2:
	month = st.text_input(
		"Month (MM) [1-12] *",
		value=str(int(_RANGES["month"][0])),
	)
with col1_3:
	year = st.text_input(
		"Year (YYYY) [2019-2022] *",
		value=str(int(_RANGES["year"][0])),
	)

col2_1, col2_2, col2_3, col2_4 = st.columns(4)
with col2_1:
	t_lockdown = st.selectbox(
		"Type of lockdown [0=no MCO, 1=MCO, 2=RMCO] *",
		options=[0, 1, 2],
		format_func=lambda v: {
			0: "0 (no MCO)",
			1: "1 (MCO)",
			2: "2 (RMCO)",
		}.get(v, str(v)),
	)
with col2_2:
	temperature = st.text_input(
		"Temperature (°C) [22-31] *",
		value=str(float(_RANGES["temperature"][0])),
	)
with col2_3:
	humidity = st.text_input(
		"Relative Humidity (%) [58-95] *",
		value=str(float(_RANGES["humidity"][0])),
	)
with col2_4:
	pressure = st.text_input(
		"Pressure (hPa) [1006-1014] *",
		value=str(float(_RANGES["pressure"][0])),
	)

col3_1, col3_2, col3_3, col3_4 = st.columns(4)
with col3_1:
	rainfall_duration = st.text_input(
		"Rainfall Duration (min) [0-60] *",
		value=str(float(_RANGES["rainfall_duration"][0])),
	)
with col3_2:
	rainfall_amount = st.text_input(
		"Rainfall amount (mm) [-14-4] *",
		value=str(0.0),
	)
with col3_3:
	wind_speed = st.text_input(
		"Wind speed (m/s) [0-5.3] *",
		value=str(float(_RANGES["wind_speed"][0])),
	)
with col3_4:
	t_day = st.selectbox(
		"Type of day [0=weekday, 1=weekend/holiday] *",
		options=[0, 1],
		format_func=lambda v: {0: "0 (weekday)", 1: "1 (weekend/holiday)"}.get(
			v, str(v)
		),
	)

st.write("")

# ==================== Buttons (Enter / Clear) ====================
_enter_validation_error = _validate_enter_inputs()
_validity = _field_validity()

_css_rules: list[str] = []
for label, is_valid in _validity.items():
	color = "green" if is_valid else "red"
	_css_rules.append(
		f"input[aria-label=\"{label}\"] {{ outline: 2px solid {color} !important; outline-offset: 2px !important; }}"
	)

st.markdown(
	"<style>" + "\n".join(_css_rules) + "</style>",
	unsafe_allow_html=True,
)

b_col1, b_col2, b_col3, b_col4, b_col5 = st.columns(5)
with b_col2:
	btn_enter = st.button(
		"Enter",
		use_container_width=True,
		disabled=_enter_validation_error is not None,
	)
with b_col3:
	btn_clear = st.button("Clear", use_container_width=True)


def _handle_enter() -> None:
	history = _load_history_cached(st.session_state.history_path)

	day_i = int(day)
	month_i = int(month)
	year_i = int(year)
	tday_i = int(t_day)
	tlock_i = int(t_lockdown)

	_require_int_range("Day", day_i, int(_RANGES["day"][0]), int(_RANGES["day"][1]))
	_require_int_range("Month", month_i, int(_RANGES["month"][0]), int(_RANGES["month"][1]))
	_require_int_range("Year", year_i, int(_RANGES["year"][0]), int(_RANGES["year"][1]))

	if tday_i not in (0, 1):
		raise ValueError("Invalid value: Type of day must be 0 (weekday) or 1 (weekend/holiday)")
	if tlock_i not in (0, 1, 2):
		raise ValueError("Invalid value: Type of lockdown must be 0 (no MCO), 1 (MCO) or 2 (RMCO)")

	temp_f = float(temperature)
	hum_f = float(humidity)
	pres_f = float(pressure)
	rain_dur_f = float(rainfall_duration)
	rain_amt_f = float(rainfall_amount)
	wind_f = float(wind_speed)

	_require_float_range("Temperature (°C)", temp_f, _RANGES["temperature"][0], _RANGES["temperature"][1])
	_require_float_range("Relative Humidity (%)", hum_f, _RANGES["humidity"][0], _RANGES["humidity"][1])
	_require_float_range("Pressure (hPa)", pres_f, _RANGES["pressure"][0], _RANGES["pressure"][1])
	_require_float_range(
		"Rainfall Duration (min)",
		rain_dur_f,
		_RANGES["rainfall_duration"][0],
		_RANGES["rainfall_duration"][1],
	)
	_require_float_range(
		"Rainfall amount (mm)",
		rain_amt_f,
		_RANGES["rainfall_amount"][0],
		_RANGES["rainfall_amount"][1],
	)
	_require_float_range(
		"Wind speed (m/s)",
		wind_f,
		_RANGES["wind_speed"][0],
		_RANGES["wind_speed"][1],
	)

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
	_persist_state_to_disk()


def _handle_clear() -> None:
	st.session_state.data_df = st.session_state.data_df.iloc[0:0]
	st.session_state.pred_df = st.session_state.pred_df.iloc[0:0]
	st.session_state.metrics = {}
	st.session_state.status = ""
	_persist_state_to_disk()


if btn_enter:
	try:
		_handle_enter()
	except Exception as e:
		st.session_state.status = str(e)

if btn_clear:
	_handle_clear()


# This corresponds to the command_text Label in Tkinter
if _enter_validation_error:
	st.write(_enter_validation_error)
elif st.session_state.status:
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
	_persist_state_to_disk()


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

# ==================== Trend chart (Actual vs Predicted) ====================
if st.session_state.pred_df is not None and len(st.session_state.pred_df) > 0:
	chart_df = st.session_state.pred_df.copy()
	if "TimeStamp" in chart_df.columns:
		parsed_ts = pd.to_datetime(chart_df["TimeStamp"], errors="coerce")
		if parsed_ts.notna().any():
			chart_df["TimeStamp"] = parsed_ts
			chart_df = chart_df.sort_values(by="TimeStamp").dropna(subset=["TimeStamp"])
		else:
			# If parsing fails, keep original order/index for a sensible trend.
			chart_df["TimeStamp"] = pd.RangeIndex(start=1, stop=len(chart_df) + 1, step=1)

	value_cols = [
		c
		for c in ["Actual Usage Peak (kwh)", "Predicted Usage Peak (kwh)"]
		if c in chart_df.columns
	]
	if value_cols and "TimeStamp" in chart_df.columns and len(chart_df) > 0:
		st.caption("Trend")
		st.line_chart(
			chart_df.set_index("TimeStamp")[value_cols],
			use_container_width=True,
		)

# Show metrics without adding extra UI elements (simple text only)
if st.session_state.metrics:
	metrics_text = ", ".join(
		[
			f"{k}: {v:.4f}"
			for k, v in st.session_state.metrics.items()
			if k != "R2 Score"
		]
	)
	st.write(metrics_text)