from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from django.core.management.base import BaseCommand, CommandError

from energy_app.energy_forecaster import (
    FEATURE_COLUMNS,
    MODEL_PATH,
    TARGET_COLUMN,
    ForecastUnavailableError,
    build_training_frame,
    load_hourly_history,
)


class Command(BaseCommand):
    help = 'Latih model peramalan konsumsi energi per jam dari dataset historis.'

    def handle(self, *args, **options):
        try:
            hourly = load_hourly_history()
        except ForecastUnavailableError as error:
            raise CommandError(str(error)) from error

        training_data = build_training_frame(hourly)
        if len(training_data) < 500:
            raise CommandError(
                f'Hanya ada {len(training_data)} baris fitur valid; minimal 500 diperlukan.'
            )

        split_index = int(len(training_data) * 0.8)
        train_data = training_data.iloc[:split_index]
        test_data = training_data.iloc[split_index:]
        model = HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=160,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=42,
        )
        model.fit(
            train_data[list(FEATURE_COLUMNS)],
            train_data[TARGET_COLUMN],
        )
        predictions = model.predict(test_data[list(FEATURE_COLUMNS)])
        metrics = {
            'mae': float(mean_absolute_error(test_data[TARGET_COLUMN], predictions)),
            'rmse': float(np.sqrt(mean_squared_error(test_data[TARGET_COLUMN], predictions))),
            'r2': float(r2_score(test_data[TARGET_COLUMN], predictions)),
        }
        artifact = {
            'model': model,
            'feature_columns': list(FEATURE_COLUMNS),
            'metrics': metrics,
            'trained_through': hourly.last_valid_index().isoformat(),
        }

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = MODEL_PATH.with_suffix('.joblib.tmp')
        joblib.dump(artifact, temporary_path)
        temporary_path.replace(MODEL_PATH)

        self.stdout.write(self.style.SUCCESS(
            f'Model peramalan tersimpan di {MODEL_PATH}\n'
            f'Baris latih: {len(train_data):,}; baris uji: {len(test_data):,}\n'
            f'MAE: {metrics["mae"]:.3f} kW | '
            f'RMSE: {metrics["rmse"]:.3f} kW | '
            f'R²: {metrics["r2"]:.3f}'
        ))
