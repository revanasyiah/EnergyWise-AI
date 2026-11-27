from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from django.conf import settings


DATASET_PATH = Path(settings.BASE_DIR) / 'dataset' / 'household_power_consumption.csv'
MODEL_PATH = Path(settings.BASE_DIR) / 'ml' / 'hourly_energy_forecaster.joblib'
TARGET_COLUMN = 'Global_active_power'
FEATURE_COLUMNS = (
    'lag_1',
    'lag_24',
    'lag_168',
    'mean_24',
    'mean_168',
    'hour_sin',
    'hour_cos',
    'weekday_sin',
    'weekday_cos',
    'yearday_sin',
    'yearday_cos',
)


class ForecastUnavailableError(Exception):
    pass


def is_forecast_model_trained():
    return MODEL_PATH.is_file()


@lru_cache(maxsize=1)
def load_hourly_history():
    if not DATASET_PATH.is_file():
        raise ForecastUnavailableError(f'Dataset tidak ditemukan: {DATASET_PATH}')

    try:
        data = pd.read_csv(
            DATASET_PATH,
            usecols=['Date', 'Time', TARGET_COLUMN],
            na_values=['?'],
            dtype={'Date': 'string', 'Time': 'string', TARGET_COLUMN: np.float32},
        )
    except (OSError, ValueError) as error:
        raise ForecastUnavailableError(
            f'Dataset konsumsi gagal dibaca: {error}'
        ) from error

    timestamps = pd.to_datetime(
        data['Date'].str.strip() + ' ' + data['Time'].str.strip(),
        format='%d/%m/%Y %H:%M:%S',
        errors='coerce',
    )
    valid = pd.DataFrame(
        {
            'timestamp': timestamps,
            TARGET_COLUMN: pd.to_numeric(data[TARGET_COLUMN], errors='coerce'),
        }
    ).dropna()
    if valid.empty:
        raise ForecastUnavailableError(
            'Dataset tidak memiliki pengukuran waktu dan daya aktif yang valid.'
        )

    hourly = valid.groupby(valid['timestamp'].dt.floor('h'))[TARGET_COLUMN].agg(
        ['mean', 'count']
    )
    hourly_values = hourly['mean'].where(hourly['count'] >= 30)
    full_index = pd.date_range(hourly.index.min(), hourly.index.max(), freq='h')
    return hourly_values.reindex(full_index).interpolate(
        method='time',
        limit=3,
        limit_direction='both',
    )


def _calendar_features(index):
    hours = index.hour.to_numpy(dtype=np.float32)
    weekdays = index.dayofweek.to_numpy(dtype=np.float32)
    yeardays = index.dayofyear.to_numpy(dtype=np.float32)
    hour_angle = 2 * np.pi * hours / 24
    weekday_angle = 2 * np.pi * weekdays / 7
    yearday_angle = 2 * np.pi * yeardays / 365.25

    return pd.DataFrame(
        {
            'hour_sin': np.sin(hour_angle),
            'hour_cos': np.cos(hour_angle),
            'weekday_sin': np.sin(weekday_angle),
            'weekday_cos': np.cos(weekday_angle),
            'yearday_sin': np.sin(yearday_angle),
            'yearday_cos': np.cos(yearday_angle),
        },
        index=index,
    )


def build_training_frame(hourly):
    previous = hourly.shift(1)
    lags = pd.DataFrame(
        {
            'lag_1': previous,
            'lag_24': hourly.shift(24),
            'lag_168': hourly.shift(168),
            'mean_24': previous.rolling(24, min_periods=24).mean(),
            'mean_168': previous.rolling(168, min_periods=168).mean(),
        },
        index=hourly.index,
    )
    calendar = _calendar_features(hourly.index)
    target = hourly.rename(TARGET_COLUMN)
    return pd.concat([lags, calendar, target], axis=1).dropna()


@lru_cache(maxsize=1)
def load_forecast_artifact():
    if not is_forecast_model_trained():
        raise ForecastUnavailableError(
            'Model prediksi belum dilatih. Jalankan '
            '`python manage.py train_energy_forecast` terlebih dahulu.'
        )

    artifact = joblib.load(MODEL_PATH)
    if artifact.get('feature_columns') != list(FEATURE_COLUMNS):
        raise ForecastUnavailableError(
            'Fitur model prediksi tidak sesuai. Latih ulang model peramalan.'
        )
    return artifact


def _make_forecast_features(hourly, timestamp):
    window = hourly.tail(168)
    if len(window) < 168 or window.isna().any():
        raise ForecastUnavailableError(
            'Data terbaru belum memiliki 168 jam pengukuran yang cukup untuk prediksi.'
        )

    lags = pd.DataFrame(
        [
            {
                'lag_1': float(window.iloc[-1]),
                'lag_24': float(window.iloc[-24]),
                'lag_168': float(window.iloc[0]),
                'mean_24': float(window.tail(24).mean()),
                'mean_168': float(window.mean()),
            }
        ],
        index=pd.DatetimeIndex([timestamp]),
    )
    return pd.concat(
        [lags, _calendar_features(lags.index)],
        axis=1,
    )[list(FEATURE_COLUMNS)]


def forecast_energy(horizon):
    if horizon not in (1, 24):
        raise ValueError('Periode prediksi harus 1 atau 24 jam.')

    artifact = load_forecast_artifact()
    hourly = load_hourly_history().copy()
    latest = hourly.last_valid_index()
    if latest is None:
        raise ForecastUnavailableError('Data historis valid tidak ditemukan.')

    hourly = hourly.loc[:latest]
    recent_history = hourly.tail(24)
    predictions = []
    forecast_times = []
    for step in range(1, horizon + 1):
        timestamp = latest + pd.Timedelta(hours=step)
        features = _make_forecast_features(hourly, timestamp)
        prediction = max(0.0, float(artifact['model'].predict(features)[0]))
        hourly.loc[timestamp] = prediction
        forecast_times.append(timestamp)
        predictions.append(prediction)

    total_energy = sum(predictions)
    baseline_energy = (
        float(recent_history.mean())
        if horizon == 1
        else float(recent_history.sum())
    )
    change_percent = (
        ((total_energy - baseline_energy) / baseline_energy) * 100
        if baseline_energy > 0
        else None
    )

    history_labels = [timestamp.strftime('%d/%m %H:%M') for timestamp in recent_history.index]
    prediction_labels = [timestamp.strftime('%d/%m %H:%M') for timestamp in forecast_times]
    history_values = [round(float(value), 3) for value in recent_history]
    prediction_values = [round(value, 3) for value in predictions]
    chart_data = {
        'labels': history_labels + prediction_labels,
        'history': history_values + [None] * horizon,
        'prediction': (
            [None] * (len(history_values) - 1)
            + [history_values[-1]]
            + prediction_values
        ),
    }

    return {
        'horizon': horizon,
        'total_kwh': round(total_energy, 3),
        'average_kw': round(total_energy / horizon, 3),
        'change_percent': (
            round(change_percent, 1) if change_percent is not None else None
        ),
        'change_label': (
            'vs rata-rata per jam, 24 jam terakhir'
            if horizon == 1
            else 'vs total 24 jam sebelumnya'
        ),
        'data_as_of': latest.strftime('%d/%m/%Y %H:%M'),
        'mae': artifact['metrics']['mae'],
        'chart_data': chart_data,
    }
