import json
from datetime import datetime
from decimal import Decimal
from unittest.mock import patch

import numpy as np
import pandas as pd
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from .dashboard_data import (
    DashboardDataUnavailableError,
    load_dashboard_data,
)
from .energy_forecaster import ForecastUnavailableError
from .gemini_analysis import (
    GeminiAnalysisError,
    analyze_forecast_with_gemini,
    analyze_with_gemini,
)
from .models import EnergyHistoryRecord


FORECAST_RESULT = {
    'horizon': 24,
    'total_kwh': 40.823,
    'average_kw': 1.701,
    'change_percent': 30.6,
    'change_label': 'vs total 24 jam sebelumnya',
    'data_as_of': '13/12/2008 21:00',
    'mae': 0.328,
    'chart_data': {
        'labels': ['13/12 21:00', '13/12 22:00'],
        'history': [0.367, None],
        'prediction': [0.367, 1.701],
    },
}

DASHBOARD_DATA = {
    'period_start': datetime(2006, 12, 16, 17),
    'period_end': datetime(2008, 12, 13, 21),
    'total_energy_kwh': 123.45,
    'average_power_kw': 1.234,
    'peak_timestamp': datetime(2007, 1, 2, 21),
    'peak_power_kw': 4.567,
    'row_count': 3,
    'submeters': [
        {'name': f'Submeter {index}', 'total_wh': 12345.0, 'valid_readings': 3}
        for index in range(1, 4)
    ],
    'chart': {
        'monthly_labels': ['12/2006', '01/2007'],
        'monthly_power_kw': [1.12, 1.34],
        'hourly_labels': [f'{hour:02d}:00' for hour in range(24)],
        'hourly_power_kw': [0.5] * 24,
    },
}


class ApplicationRouteTests(TestCase):
    pages = (
        ('dashboard', '/', 'Selamat datang di EnergyWise AI', 'Dasbor'),
        ('estimation', '/estimation/', 'Estimasi energi', 'Estimasi'),
        ('prediction', '/prediction/', 'Prediksi konsumsi energi', 'Prediksi'),
        ('cost', '/cost/', 'Estimasi biaya listrik', 'Estimasi Biaya'),
        ('history', '/history/', 'Riwayat prediksi', 'Riwayat Prediksi'),
    )

    @patch(
        'energy_app.views.load_dashboard_data',
        side_effect=DashboardDataUnavailableError('Dataset tidak ditemukan.'),
    )
    def test_navigation_routes_render_their_pages(self, load_dashboard_data_mock):
        for route_name, expected_path, heading, _ in self.pages:
            with self.subTest(route=route_name):
                path = reverse(route_name)
                self.assertEqual(path, expected_path)

                response = self.client.get(path)

                self.assertEqual(response.status_code, 200)
                self.assertContains(response, heading)
                self.assertContains(response, '<html lang="id">')
                for linked_route, _, _, navigation_label in self.pages:
                    self.assertContains(
                        response,
                        f'href="{reverse(linked_route)}"',
                    )
                    self.assertContains(response, navigation_label)

    @patch('energy_app.views.load_dashboard_data', return_value=DASHBOARD_DATA)
    def test_dashboard_displays_dataset_metrics_and_charts(
        self,
        load_dashboard_data_mock,
    ):
        response = self.client.get(reverse('dashboard'))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Dataset terhubung')
        self.assertContains(response, '123,45')
        self.assertContains(response, '1,234')
        self.assertContains(response, '21:00')
        self.assertContains(response, '12.345')
        self.assertContains(response, 'dashboard-chart-data')
        self.assertContains(response, 'monthly_power_kw')
        self.assertContains(response, 'TARIF BELUM DIATUR')

    @patch(
        'energy_app.views.load_dashboard_data',
        side_effect=DashboardDataUnavailableError('Dataset tidak ditemukan.'),
    )
    def test_dashboard_explains_when_dataset_cannot_be_loaded(
        self,
        load_dashboard_data_mock,
    ):
        response = self.client.get(reverse('dashboard'))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Dataset tidak terhubung')
        self.assertContains(response, 'Dataset tidak ditemukan.')
        self.assertContains(response, 'Belum ada data pengukuran')

    def test_form_validation_uses_indonesian_messages(self):
        response = self.client.post(reverse('estimation'), {})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Bidang ini tidak boleh kosong.')
        self.assertNotContains(response, 'This field is required.')

    def test_workspace_control_explains_its_purpose_and_has_shortcuts(self):
        response = self.client.get(reverse('history'))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'aria-label="Menu ruang kerja: Rumah tangga saya"')
        self.assertContains(response, 'aria-expanded="false"')
        self.assertContains(response, 'aria-controls="workspace-menu"')
        self.assertContains(response, 'bukan akun atau profil pengguna')
        self.assertContains(response, 'Dasbor energi')
        self.assertContains(response, 'Riwayat prediksi')

    def test_cost_amounts_use_readable_indonesian_currency_format(self):
        response = self.client.post(
            reverse('cost'),
            {
                'power_kw': '2',
                'hours_per_day': '10',
                'tariff_per_kwh': '14500',
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '8.700.000,00')
        self.assertContains(response, '290.000,00')
        self.assertContains(response, '14.500,00')
        self.assertNotContains(response, '8700000,00')

    @patch('energy_app.views.is_model_trained', return_value=True)
    @patch(
        'energy_app.views.predict_active_power',
        return_value={'prediction': 1.234, 'mae': 0.2},
    )
    def test_estimation_shows_model_result_after_valid_input(
        self,
        predict_active_power,
        is_model_trained,
    ):
        response = self.client.post(
            reverse('estimation'),
            {
                'global_reactive_power': '0.120',
                'voltage': '240.00',
                'sub_metering_1': '1',
                'sub_metering_2': '2',
                'sub_metering_3': '3',
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '1,234')
        self.assertContains(response, 'HASIL MODEL')
        self.assertContains(response, 'MAE pada data uji: 0,200 kW.')
        predict_active_power.assert_called_once()
        self.assertEqual(
            EnergyHistoryRecord.objects.get().prediction_type,
            EnergyHistoryRecord.RecordType.ESTIMATION,
        )

    @patch('energy_app.views.is_model_trained', return_value=False)
    @patch(
        'energy_app.views.predict_active_power',
        side_effect=FileNotFoundError(
            'Model belum tersedia. Jalankan `python manage.py train_energy_model` terlebih dahulu.'
        ),
    )
    def test_estimation_explains_how_to_train_missing_model(
        self,
        predict_active_power,
        is_model_trained,
    ):
        response = self.client.post(
            reverse('estimation'),
            {
                'global_reactive_power': '0.120',
                'voltage': '240.00',
                'sub_metering_1': '1',
                'sub_metering_2': '2',
                'sub_metering_3': '3',
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'python manage.py train_energy_model')
        self.assertContains(response, '— <span>kW</span>')

    @override_settings(GEMINI_API_KEY='test-api-key')
    @patch('energy_app.views.is_model_trained', return_value=True)
    @patch(
        'energy_app.views.predict_active_power',
        return_value={'prediction': 1.234, 'mae': 0.2},
    )
    @patch(
        'energy_app.views.analyze_with_gemini',
        return_value={
            'summary': 'Estimasi ini merupakan perkiraan.',
            'input_analysis': 'Model menggunakan seluruh nilai masukan.',
            'model_accuracy': 'MAE adalah rata-rata galat pada data uji.',
            'recommendation': 'Pantau penggunaan listrik secara berkala.',
        },
    )
    def test_ai_analysis_button_uses_gemini(
        self,
        analyze_with_gemini,
        predict_active_power,
        is_model_trained,
    ):
        response = self.client.post(
            reverse('estimation'),
            {
                'action': 'analyze',
                'global_reactive_power': '0.120',
                'voltage': '240.00',
                'sub_metering_1': '1',
                'sub_metering_2': '2',
                'sub_metering_3': '3',
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Analisis AI')
        self.assertContains(response, 'Analisis Gemini')
        self.assertContains(response, 'Estimasi ini merupakan perkiraan.')
        self.assertContains(response, 'Analisis masukan')
        self.assertContains(response, 'Pantau penggunaan listrik secara berkala.')
        self.assertContains(response, 'dikirim ke Google Gemini')
        analyze_with_gemini.assert_called_once_with(
            {
                'global_reactive_power': Decimal('0.120'),
                'voltage': Decimal('240.00'),
                'sub_metering_1': Decimal('1'),
                'sub_metering_2': Decimal('2'),
                'sub_metering_3': Decimal('3'),
            },
            1.234,
            0.2,
        )

    @override_settings(GEMINI_API_KEY='')
    @patch('energy_app.views.is_model_trained', return_value=True)
    @patch(
        'energy_app.views.predict_active_power',
        return_value={'prediction': 1.234, 'mae': 0.2},
    )
    def test_gemini_button_is_hidden_without_api_key(
        self,
        predict_active_power,
        is_model_trained,
    ):
        response = self.client.post(
            reverse('estimation'),
            {
                'global_reactive_power': '0.120',
                'voltage': '240.00',
                'sub_metering_1': '1',
                'sub_metering_2': '2',
                'sub_metering_3': '3',
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Atur GEMINI_API_KEY')
        self.assertNotContains(response, 'Analisis AI')


class HistoryDetailViewTests(TestCase):
    def setUp(self):
        self.record = EnergyHistoryRecord.objects.create(
            prediction_type=EnergyHistoryRecord.RecordType.PREDICTION,
            input_summary='Periode 24 jam; data hingga 13/12/2008 21:00',
            result_value=Decimal('40.823'),
            result_unit='kWh',
            details=(
                'Daya rata-rata: 1,701 kW; perubahan dari histori: +30,6%; '
                'MAE uji: 0,328 kW.'
            ),
        )

    def test_history_rincian_link_opens_structured_detail_page(self):
        history_response = self.client.get(reverse('history'))

        self.assertContains(
            history_response,
            reverse('history_detail', args=(self.record.pk,)),
        )
        self.assertContains(history_response, 'Rincian')

        detail_response = self.client.get(
            reverse('history_detail', args=(self.record.pk,))
        )

        self.assertEqual(detail_response.status_code, 200)
        self.assertContains(detail_response, 'Rincian catatan')
        self.assertContains(detail_response, '40,823')
        self.assertContains(detail_response, 'Periode 24 jam')
        self.assertContains(detail_response, 'MAE uji')
        self.assertContains(detail_response, 'Prediksi baru')
        self.assertNotContains(detail_response, '<details')

    def test_missing_history_record_returns_404(self):
        response = self.client.get(
            reverse('history_detail', args=(self.record.pk + 1,))
        )

        self.assertEqual(response.status_code, 404)


class EnergyHistoryAdminTests(TestCase):
    def setUp(self):
        self.admin_user = get_user_model().objects.create_superuser(
            username='energy-admin',
            email='admin@example.com',
            password='test-admin-password',
        )
        self.client.force_login(self.admin_user)
        self.record = EnergyHistoryRecord.objects.create(
            prediction_type=EnergyHistoryRecord.RecordType.PREDICTION,
            input_summary='Periode 24 jam; data hingga 13/12/2008 21:00',
            result_value=Decimal('40.823'),
            result_unit='kWh',
            details='MAE uji: 0,328 kW.',
        )

    def test_admin_home_shows_localized_history_model(self):
        response = self.client.get(reverse('admin:index'))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Pengelolaan data energi')
        self.assertContains(response, 'Data energi')
        self.assertContains(response, 'Riwayat energi')
        self.assertContains(response, 'Database aplikasi')
        self.assertContains(response, 'SQLite')
        self.assertContains(response, 'Tabel riwayat energi')
        self.assertContains(response, 'Jumlah catatan tersimpan')
        self.assertContains(response, 'Riwayat energi terbaru')
        self.assertContains(response, 'Periode 24 jam')
        self.assertEqual(response.context['energywise_history_count'], 1)
        self.assertContains(
            response,
            reverse('admin:energy_app_energyhistoryrecord_changelist'),
        )

    def test_energy_history_is_listed_and_searchable_in_admin(self):
        response = self.client.get(
            reverse('admin:energy_app_energyhistoryrecord_changelist'),
            {'q': '13/12/2008'},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Administrasi EnergyWise AI')
        self.assertContains(response, 'Periode 24 jam')
        self.assertContains(response, '40.823 kWh')
        self.assertContains(response, 'Waktu')
        self.assertContains(response, 'Jenis')

        change_response = self.client.get(
            reverse(
                'admin:energy_app_energyhistoryrecord_change',
                args=(self.record.pk,),
            )
        )
        self.assertContains(change_response, 'Catatan energi')
        self.assertContains(change_response, 'MAE uji: 0,328 kW.')


class DashboardDataTests(TestCase):
    def setUp(self):
        load_dashboard_data.cache_clear()

    def tearDown(self):
        load_dashboard_data.cache_clear()

    @patch('energy_app.dashboard_data.pd.read_csv')
    @patch('energy_app.dashboard_data.load_hourly_history')
    def test_dashboard_data_aggregates_hourly_and_submeter_readings(
        self,
        load_hourly_history_mock,
        read_csv_mock,
    ):
        timestamps = pd.to_datetime(
            [
                '2024-01-01 10:00',
                '2024-01-01 11:00',
                '2024-01-01 12:00',
            ]
        )
        load_hourly_history_mock.return_value = pd.Series(
            [1.0, 2.0, 3.0],
            index=timestamps,
        )
        read_csv_mock.return_value = [
            pd.DataFrame(
                {
                    'Sub_metering_1': [1.0, np.nan],
                    'Sub_metering_2': [2.0, 3.0],
                    'Sub_metering_3': [0.0, 5.0],
                }
            )
        ]

        dataset = load_dashboard_data()

        self.assertEqual(dataset['total_energy_kwh'], 6.0)
        self.assertEqual(dataset['average_power_kw'], 2.0)
        self.assertEqual(dataset['peak_timestamp'], timestamps[-1])
        self.assertEqual(dataset['row_count'], 2)
        self.assertEqual(
            [meter['total_wh'] for meter in dataset['submeters']],
            [1.0, 5.0, 5.0],
        )
        self.assertEqual(dataset['chart']['monthly_power_kw'], [2.0])
        self.assertEqual(
            dataset['chart']['hourly_power_kw'][10:13],
            [1.0, 2.0, 3.0],
        )

    @patch(
        'energy_app.dashboard_data.load_hourly_history',
        side_effect=ForecastUnavailableError('Dataset tidak ditemukan.'),
    )
    def test_dashboard_data_surfaces_missing_dataset(
        self,
        load_hourly_history_mock,
    ):
        with self.assertRaisesMessage(
            DashboardDataUnavailableError,
            'Dataset tidak ditemukan.',
        ):
            load_dashboard_data()


class GeminiAnalysisServiceTests(TestCase):
    @override_settings(
        GEMINI_API_KEY='test-api-key',
        GEMINI_MODEL='test-gemini-model',
        GEMINI_FALLBACK_MODEL='',
    )
    @patch('energy_app.gemini_analysis.genai.Client')
    def test_analysis_uses_configured_gemini_model_and_closes_client(
        self,
        client_factory,
    ):
        client = client_factory.return_value
        client.models.generate_content.return_value.text = (
            json.dumps(
                {
                    'summary': 'Estimasi ini merupakan perkiraan.',
                    'input_analysis': 'Semua masukan telah dianalisis.',
                    'model_accuracy': 'MAE adalah rata-rata galat data uji.',
                    'recommendation': 'Pantau penggunaan listrik.',
                }
            )
        )

        result = analyze_with_gemini(
            {
                'global_reactive_power': Decimal('0.120'),
                'voltage': Decimal('240.00'),
                'sub_metering_1': Decimal('1'),
                'sub_metering_2': Decimal('2'),
                'sub_metering_3': Decimal('3'),
            },
            1.234,
            0.2,
        )

        self.assertEqual(
            result['summary'],
            'Estimasi ini merupakan perkiraan.',
        )
        request = client.models.generate_content.call_args.kwargs
        self.assertEqual(request['model'], 'test-gemini-model')
        self.assertIn('"estimasi_daya_aktif_kw": 1.234', request['contents'])
        self.assertEqual(
            request['config'].response_mime_type,
            'application/json',
        )
        self.assertEqual(request['config'].max_output_tokens, 1400)
        client_factory.assert_called_once_with(api_key='test-api-key')
        client.close.assert_called_once_with()

    @override_settings(
        GEMINI_API_KEY='test-api-key',
        GEMINI_MODEL='test-gemini-model',
        GEMINI_FALLBACK_MODEL='',
    )
    @patch('energy_app.gemini_analysis.genai.Client')
    def test_forecast_analysis_uses_chart_series_and_validates_fields(
        self,
        client_factory,
    ):
        client = client_factory.return_value
        client.models.generate_content.return_value.text = json.dumps(
            {
                'summary': 'Proyeksi 24 jam mencapai 40,823 kWh.',
                'historical_pattern': 'Daya historis tercatat 0,367 kW.',
                'forecast_pattern': 'Prediksi per jam menunjukkan 1,701 kW.',
                'model_accuracy': 'MAE uji model sebesar 0,328 kW.',
                'recommendation': 'Bandingkan pola ini dengan pemakaian aktual.',
            }
        )

        result = analyze_forecast_with_gemini(FORECAST_RESULT)

        self.assertEqual(result['summary'], 'Proyeksi 24 jam mencapai 40,823 kWh.')
        prompt = client.models.generate_content.call_args.kwargs['contents']
        self.assertIn('"dataset_latest_observation": "13/12/2008 21:00"', prompt)
        self.assertIn('"forecast_total_energy_kwh": 40.823', prompt)
        prompt_data = json.loads(prompt.rsplit('\n', 1)[-1])
        self.assertEqual(
            prompt_data['predicted_hourly_power_kw'],
            [{'time': '13/12 22:00', 'power_kw': 1.701}],
        )
        client.close.assert_called_once_with()


class PredictionViewTests(TestCase):
    @patch('energy_app.views.is_forecast_model_trained', return_value=True)
    @patch(
        'energy_app.views.forecast_energy',
        return_value=FORECAST_RESULT,
    )
    def test_prediction_displays_forecast_and_chart_data(
        self,
        forecast_energy,
        is_forecast_model_trained,
    ):
        response = self.client.post(
            reverse('prediction'),
            {'horizon': '24'},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '40,823')
        self.assertContains(response, '+30,6%')
        self.assertContains(response, '13/12/2008 21:00')
        self.assertContains(response, 'Hasil prediksi disimpan di Riwayat Prediksi')
        self.assertContains(response, 'forecast-chart-data')
        self.assertEqual(forecast_energy.call_args.args, (24,))
        record = EnergyHistoryRecord.objects.get()
        self.assertEqual(
            record.prediction_type,
            EnergyHistoryRecord.RecordType.PREDICTION,
        )
        self.assertEqual(record.result_value, Decimal('40.823'))
        self.assertIn('Periode 24 jam', record.input_summary)
        history_response = self.client.get(reverse('history'))
        self.assertContains(history_response, '40,823 kWh')
        self.assertContains(history_response, '13/12/2008 21:00')

    @override_settings(GEMINI_API_KEY='test-api-key')
    @patch('energy_app.views.is_forecast_model_trained', return_value=True)
    @patch('energy_app.views.forecast_energy', return_value=FORECAST_RESULT)
    @patch(
        'energy_app.views.analyze_forecast_with_gemini',
        return_value={
            'summary': 'Prediksi energi untuk 24 jam.',
            'historical_pattern': 'Pola historis terlihat stabil.',
            'forecast_pattern': 'Daya diprediksi meningkat.',
            'model_accuracy': 'MAE uji sebesar 0,328 kW.',
            'recommendation': 'Pantau jam dengan daya tertinggi.',
        },
    )
    def test_forecast_ai_analysis_is_shown_on_the_chart(
        self,
        analyze_forecast_with_gemini,
        forecast_energy,
        is_forecast_model_trained,
    ):
        response = self.client.post(
            reverse('prediction'),
            {'horizon': '24', 'action': 'analyze'},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Analisis AI Gemini')
        self.assertContains(response, 'Pola historis terlihat stabil.')
        self.assertContains(response, 'Ringkasan data grafik dikirim ke Google Gemini')
        analyze_forecast_with_gemini.assert_called_once_with(FORECAST_RESULT)
        forecast_energy.assert_called_once_with(24)
        self.assertEqual(EnergyHistoryRecord.objects.count(), 0)

    @patch('energy_app.views.is_forecast_model_trained', return_value=False)
    @patch(
        'energy_app.views.forecast_energy',
        side_effect=ForecastUnavailableError(
            'Model prediksi belum dilatih. Jalankan `python manage.py train_energy_forecast` terlebih dahulu.'
        ),
    )
    def test_prediction_explains_how_to_train_missing_model(
        self,
        forecast_energy,
        is_forecast_model_trained,
    ):
        response = self.client.post(
            reverse('prediction'),
            {'horizon': '1'},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'python manage.py train_energy_forecast')
        self.assertContains(response, '— <span>kWh</span>')

    @override_settings(GEMINI_API_KEY='')
    def test_analysis_fails_clearly_when_api_key_is_missing(self):
        with self.assertRaisesMessage(
            GeminiAnalysisError,
            'GEMINI_API_KEY belum diatur pada file .env.',
        ):
            analyze_with_gemini({}, 1.234, 0.2)

    @override_settings(
        GEMINI_API_KEY='test-api-key',
        GEMINI_MODEL='test-gemini-model',
        GEMINI_FALLBACK_MODEL='',
    )
    @patch('energy_app.gemini_analysis.genai.Client')
    def test_analysis_rejects_truncated_or_invalid_gemini_response(
        self,
        client_factory,
    ):
        client = client_factory.return_value
        client.models.generate_content.return_value.text = (
            '{"summary":"Estimasi","input_analysis":'
        )

        with self.assertRaisesMessage(
            GeminiAnalysisError,
            'Respons Gemini belum lengkap atau tidak sesuai format.',
        ):
            analyze_with_gemini(
                {
                    'global_reactive_power': Decimal('0.120'),
                    'voltage': Decimal('240.00'),
                    'sub_metering_1': Decimal('1'),
                    'sub_metering_2': Decimal('2'),
                    'sub_metering_3': Decimal('3'),
                },
                1.234,
                0.2,
            )

        client.close.assert_called_once_with()
