from functools import lru_cache

import numpy as np
import pandas as pd

from .energy_forecaster import (
    DATASET_PATH,
    ForecastUnavailableError,
    load_hourly_history,
)


SUBMETER_COLUMNS = (
    'Sub_metering_1',
    'Sub_metering_2',
    'Sub_metering_3',
)


class DashboardDataUnavailableError(Exception):
    pass


@lru_cache(maxsize=1)
def load_dashboard_data():
    try:
        hourly = load_hourly_history().dropna()
    except ForecastUnavailableError as error:
        raise DashboardDataUnavailableError(str(error)) from error

    if hourly.empty:
        raise DashboardDataUnavailableError(
            'Dataset tidak memiliki data daya aktif yang dapat ditampilkan.'
        )

    meter_totals = np.zeros(len(SUBMETER_COLUMNS), dtype=np.float64)
    meter_readings = np.zeros(len(SUBMETER_COLUMNS), dtype=np.int64)
    row_count = 0
    try:
        chunks = pd.read_csv(
            DATASET_PATH,
            usecols=list(SUBMETER_COLUMNS),
            na_values=['?'],
            dtype={column: np.float64 for column in SUBMETER_COLUMNS},
            chunksize=100_000,
        )
        for chunk in chunks:
            row_count += len(chunk)
            for index, column in enumerate(SUBMETER_COLUMNS):
                values = chunk[column]
                meter_totals[index] += float(values.sum(skipna=True))
                meter_readings[index] += int(values.count())
    except (OSError, ValueError) as error:
        raise DashboardDataUnavailableError(
            f'Data submeter pada dataset gagal dibaca: {error}'
        ) from error

    monthly_power = hourly.resample('MS').mean().dropna()
    hourly_profile = hourly.groupby(hourly.index.hour).mean().reindex(range(24))
    peak_timestamp = hourly.idxmax()

    return {
        'period_start': hourly.index[0],
        'period_end': hourly.index[-1],
        'total_energy_kwh': float(hourly.sum()),
        'average_power_kw': float(hourly.mean()),
        'peak_timestamp': peak_timestamp,
        'peak_power_kw': float(hourly.loc[peak_timestamp]),
        'row_count': row_count,
        'submeters': [
            {
                'name': f'Submeter {index + 1}',
                'total_wh': (
                    float(meter_totals[index])
                    if meter_readings[index]
                    else None
                ),
                'valid_readings': int(meter_readings[index]),
            }
            for index in range(len(SUBMETER_COLUMNS))
        ],
        'chart': {
            'monthly_labels': [
                timestamp.strftime('%m/%Y')
                for timestamp in monthly_power.index
            ],
            'monthly_power_kw': [
                round(float(value), 3) for value in monthly_power
            ],
            'hourly_labels': [
                f'{hour:02d}:00' for hour in hourly_profile.index
            ],
            'hourly_power_kw': [
                round(float(value), 3) if pd.notna(value) else None
                for value in hourly_profile
            ],
        },
    }
