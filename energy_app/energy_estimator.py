from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from django.conf import settings


FEATURE_COLUMNS = (
    'Global_reactive_power',
    'Voltage',
    'Sub_metering_1',
    'Sub_metering_2',
    'Sub_metering_3',
)
FORM_FIELDS = (
    'global_reactive_power',
    'voltage',
    'sub_metering_1',
    'sub_metering_2',
    'sub_metering_3',
)
MODEL_PATH = Path(settings.BASE_DIR) / 'ml' / 'active_power_estimator.joblib'


def is_model_trained():
    return MODEL_PATH.is_file()


@lru_cache(maxsize=1)
def load_model_artifact():
    if not is_model_trained():
        raise FileNotFoundError(
            'Model belum tersedia. Jalankan '
            '`python manage.py train_energy_model` terlebih dahulu.'
        )

    artifact = joblib.load(MODEL_PATH)
    if artifact.get('feature_columns') != list(FEATURE_COLUMNS):
        raise ValueError('Urutan fitur model tidak sesuai dengan fitur estimasi.')
    return artifact


def predict_active_power(inputs):
    artifact = load_model_artifact()
    features = np.array(
        [[float(inputs[field]) for field in FORM_FIELDS]],
        dtype=np.float32,
    )
    prediction = float(artifact['model'].predict(features)[0])
    return {
        'prediction': max(0.0, prediction),
        'mae': artifact['metrics']['mae'],
    }
