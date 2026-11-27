from decimal import Decimal, ROUND_HALF_UP

from django.shortcuts import get_object_or_404, render
from django.utils.formats import date_format, number_format

from .dashboard_data import DashboardDataUnavailableError, load_dashboard_data
from .energy_estimator import (
    is_model_trained,
    predict_active_power,
)
from .energy_forecaster import (
    ForecastUnavailableError,
    forecast_energy,
    is_forecast_model_trained,
)
from .forms import CostEstimateForm, EstimationInputForm, PredictionHorizonForm
from .models import EnergyHistoryRecord


def dashboard(request):
    try:
        dataset = load_dashboard_data()
    except DashboardDataUnavailableError as error:
        return render(
            request,
            'energy_app/dashboard.html',
            {
                'dashboard_data': None,
                'dataset_error': str(error),
                'meter_summaries': [],
            },
        )

    peak_timestamp = dataset['peak_timestamp']
    meter_summaries = [
        {
            **meter,
            'value_display': (
                number_format(
                    meter['total_wh'],
                    decimal_pos=0,
                    force_grouping=True,
                )
                if meter['total_wh'] is not None
                else None
            ),
        }
        for meter in dataset['submeters']
    ]
    return render(
        request,
        'energy_app/dashboard.html',
        {
            'dashboard_data': dataset,
            'dataset_error': None,
            'dataset_period': (
                f'{date_format(dataset["period_start"], "d/m/Y")} – '
                f'{date_format(dataset["period_end"], "d/m/Y")}'
            ),
            'total_energy_display': number_format(
                dataset['total_energy_kwh'],
                decimal_pos=2,
                force_grouping=True,
            ),
            'average_power_display': number_format(
                dataset['average_power_kw'],
                decimal_pos=3,
            ),
            'peak_hour_display': date_format(peak_timestamp, 'H:i'),
            'peak_note': (
                f'Daya rata-rata {number_format(dataset["peak_power_kw"], decimal_pos=3)} kW '
                f'pada {date_format(peak_timestamp, "d/m/Y")}'
            ),
            'row_count_display': number_format(
                dataset['row_count'],
                decimal_pos=0,
                force_grouping=True,
            ),
            'meter_summaries': meter_summaries,
            'chart_data': dataset['chart'],
        },
    )


def estimation(request):
    form_data = request.POST if request.method == 'POST' else None
    form = EstimationInputForm(form_data)
    input_summary = None
    status_message = None
    estimate_result = None
    model_ready = is_model_trained()

    if request.method == 'POST' and form.is_valid():
        input_summary = form.cleaned_data
        try:
            result = predict_active_power(input_summary)
        except FileNotFoundError as error:
            status_message = str(error)
        else:
            estimate_result = result['prediction']
            EnergyHistoryRecord.objects.create(
                prediction_type=EnergyHistoryRecord.RecordType.ESTIMATION,
                input_summary=(
                    f'Reaktif {number_format(input_summary["global_reactive_power"], decimal_pos=3)} kW; '
                    f'tegangan {number_format(input_summary["voltage"], decimal_pos=2)} V; '
                    f'submeter {number_format(input_summary["sub_metering_1"], decimal_pos=2)}, '
                    f'{number_format(input_summary["sub_metering_2"], decimal_pos=2)}, '
                    f'{number_format(input_summary["sub_metering_3"], decimal_pos=2)}'
                ),
                result_value=Decimal(str(estimate_result)),
                result_unit='kW',
                details=(
                    f'MAE model pada data uji: '
                    f'{number_format(result["mae"], decimal_pos=3)} kW.'
                ),
            )
            status_message = (
                f'Estimasi berhasil dibuat oleh model. '
                f'MAE pada data uji: {number_format(result["mae"], decimal_pos=3)} kW. '
                'Hasil estimasi disimpan di Riwayat Prediksi.'
            )

    return render(
        request,
        'energy_app/estimation.html',
        {
            'form': form,
            'input_summary': input_summary,
            'status_message': status_message,
            'estimate_result': estimate_result,
            'model_ready': model_ready,
        },
    )


def prediction(request):
    form_data = request.POST if request.method == 'POST' else None
    form = PredictionHorizonForm(form_data)
    selected_horizon = form['horizon'].value()
    status_message = None
    forecast_result = None
    forecast_model_ready = is_forecast_model_trained()

    if request.method == 'POST' and form.is_valid():
        selected_horizon = int(form.cleaned_data['horizon'])
        try:
            forecast_result = forecast_energy(selected_horizon)
        except ForecastUnavailableError as error:
            status_message = str(error)
        else:
            status_message = (
                f'Prediksi dibuat dari data historis terakhir pada '
                f'{forecast_result["data_as_of"]}; hasil ini bukan pembacaan rumah '
                f'secara real-time.'
            )
            change_percent = forecast_result['change_percent']
            change_text = (
                f'{number_format(abs(change_percent), decimal_pos=1)}%'
                if change_percent is not None
                else 'tidak tersedia'
            )
            if change_percent is not None and change_percent > 0:
                change_text = f'+{change_text}'
            elif change_percent is not None and change_percent < 0:
                change_text = f'-{change_text}'

            EnergyHistoryRecord.objects.create(
                prediction_type=EnergyHistoryRecord.RecordType.PREDICTION,
                input_summary=(
                    f'Periode {selected_horizon} jam; data hingga '
                    f'{forecast_result["data_as_of"]}'
                ),
                result_value=Decimal(str(forecast_result['total_kwh'])),
                result_unit='kWh',
                details=(
                    f'Daya rata-rata: '
                    f'{number_format(forecast_result["average_kw"], decimal_pos=3)} kW; '
                    f'perubahan dari histori: {change_text}; '
                    f'MAE uji: {number_format(forecast_result["mae"], decimal_pos=3)} kW.'
                ),
            )
            status_message += ' Hasil prediksi disimpan di Riwayat Prediksi.'

    return render(
        request,
        'energy_app/prediction.html',
        {
            'form': form,
            'selected_horizon': selected_horizon,
            'status_message': status_message,
            'forecast_result': forecast_result,
            'forecast_model_ready': forecast_model_ready,
            'chart_data': (
                forecast_result['chart_data'] if forecast_result else None
            ),
        },
    )


def cost(request):
    form_data = request.POST if request.method == 'POST' else None
    form = CostEstimateForm(form_data)
    calculation = None

    if request.method == 'POST' and form.is_valid():
        values = form.cleaned_data
        daily_energy = values['power_kw'] * values['hours_per_day']
        monthly_energy = daily_energy * Decimal('30')
        calculation = {
            'power_kw': values['power_kw'],
            'hours_per_day': values['hours_per_day'],
            'tariff_per_kwh': values['tariff_per_kwh'],
            'daily_energy_kwh': daily_energy.quantize(
                Decimal('0.001'), rounding=ROUND_HALF_UP
            ),
            'monthly_energy_kwh': monthly_energy.quantize(
                Decimal('0.001'), rounding=ROUND_HALF_UP
            ),
            'daily_cost': (daily_energy * values['tariff_per_kwh']).quantize(
                Decimal('0.01'), rounding=ROUND_HALF_UP
            ),
            'monthly_cost': (
                monthly_energy * values['tariff_per_kwh']
            ).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
        }

    return render(
        request,
        'energy_app/cost.html',
        {'form': form, 'calculation': calculation},
    )


def history(request):
    return render(
        request,
        'energy_app/history.html',
        {'history_records': EnergyHistoryRecord.objects.all()},
    )


def history_detail(request, pk):
    record = get_object_or_404(EnergyHistoryRecord, pk=pk)
    input_items = [
        item.strip()
        for item in record.input_summary.split(';')
        if item.strip()
    ]
    detail_items = []
    for item in record.details.split(';'):
        label, separator, value = item.partition(':')
        if separator:
            detail_items.append(
                {'label': label.strip(), 'value': value.strip().rstrip('.')}
            )
        elif item.strip():
            detail_items.append({'label': 'Catatan', 'value': item.strip()})

    return render(
        request,
        'energy_app/history_detail.html',
        {
            'record': record,
            'input_items': input_items,
            'detail_items': detail_items,
            'result_display': number_format(
                record.result_value,
                decimal_pos=3,
            ),
        },
    )
