from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from energy_app.energy_estimator import FEATURE_COLUMNS, MODEL_PATH


class Command(BaseCommand):
    help = 'Latih model estimasi daya aktif dari dataset konsumsi rumah tangga.'

    def handle(self, *args, **options):
        dataset_path = Path(settings.BASE_DIR) / 'dataset' / 'household_power_consumption.csv'
        if not dataset_path.is_file():
            raise CommandError(f'Dataset tidak ditemukan: {dataset_path}')

        columns = list(FEATURE_COLUMNS) + ['Global_active_power']
        try:
            data = pd.read_csv(
                dataset_path,
                usecols=columns,
                na_values=['?'],
                dtype={column: np.float32 for column in columns},
            ).dropna()
        except (OSError, ValueError) as error:
            raise CommandError(f'Dataset gagal dibaca: {error}') from error

        if len(data) < 100:
            raise CommandError(
                f'Dataset hanya memiliki {len(data)} baris valid; minimal 100 diperlukan.'
            )

        x = data[list(FEATURE_COLUMNS)].to_numpy()
        y = data['Global_active_power'].to_numpy()
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=42,
        )

        model = HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_iter=120,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=42,
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        metrics = {
            'mae': float(mean_absolute_error(y_test, predictions)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, predictions))),
            'r2': float(r2_score(y_test, predictions)),
        }

        artifact = {
            'model': model,
            'feature_columns': list(FEATURE_COLUMNS),
            'metrics': metrics,
        }
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = MODEL_PATH.with_suffix('.joblib.tmp')
        joblib.dump(artifact, temporary_path)
        temporary_path.replace(MODEL_PATH)

        self.stdout.write(self.style.SUCCESS(
            f'Model tersimpan di {MODEL_PATH}\n'
            f'Data latih: {len(x_train):,} baris; data uji: {len(x_test):,} baris\n'
            f'MAE: {metrics["mae"]:.3f} kW | '
            f'RMSE: {metrics["rmse"]:.3f} kW | '
            f'R²: {metrics["r2"]:.3f}'
        ))
