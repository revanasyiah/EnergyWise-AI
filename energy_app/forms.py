from django import forms


class EstimationInputForm(forms.Form):
    global_reactive_power = forms.DecimalField(
        label='Daya Reaktif Global (kW)',
        min_value=0,
        max_digits=10,
        decimal_places=3,
        widget=forms.NumberInput(
            attrs={
                'class': 'form-control',
                'step': '0.001',
                'placeholder': 'mis. 0.120',
                'inputmode': 'decimal',
            }
        ),
    )
    voltage = forms.DecimalField(
        label='Tegangan (V)',
        min_value=0.01,
        max_digits=10,
        decimal_places=2,
        widget=forms.NumberInput(
            attrs={
                'class': 'form-control',
                'step': '0.01',
                'placeholder': 'mis. 240.00',
                'inputmode': 'decimal',
            }
        ),
    )
    sub_metering_1 = forms.DecimalField(
        label='Submeter 1 (Wh)',
        min_value=0,
        max_digits=10,
        decimal_places=2,
        widget=forms.NumberInput(
            attrs={
                'class': 'form-control',
                'step': '0.01',
                'placeholder': 'Masukkan nilai meter',
                'inputmode': 'decimal',
            }
        ),
    )
    sub_metering_2 = forms.DecimalField(
        label='Submeter 2 (Wh)',
        min_value=0,
        max_digits=10,
        decimal_places=2,
        widget=forms.NumberInput(
            attrs={
                'class': 'form-control',
                'step': '0.01',
                'placeholder': 'Masukkan nilai meter',
                'inputmode': 'decimal',
            }
        ),
    )
    sub_metering_3 = forms.DecimalField(
        label='Submeter 3 (Wh)',
        min_value=0,
        max_digits=10,
        decimal_places=2,
        widget=forms.NumberInput(
            attrs={
                'class': 'form-control',
                'step': '0.01',
                'placeholder': 'Masukkan nilai meter',
                'inputmode': 'decimal',
            }
        ),
    )


class PredictionHorizonForm(forms.Form):
    horizon = forms.ChoiceField(
        label='Periode prediksi',
        choices=(('1', '1 jam ke depan'), ('24', '24 jam ke depan')),
        initial='1',
        widget=forms.Select(attrs={'class': 'form-select'}),
    )


class CostEstimateForm(forms.Form):
    power_kw = forms.DecimalField(
        label='Daya aktif rata-rata (kW)',
        min_value=0,
        max_digits=10,
        decimal_places=3,
        widget=forms.NumberInput(
            attrs={
                'class': 'form-control',
                'step': '0.001',
                'placeholder': 'mis. 1.250',
                'inputmode': 'decimal',
            }
        ),
    )
    hours_per_day = forms.DecimalField(
        label='Lama penggunaan per hari (jam)',
        min_value=0,
        max_value=24,
        max_digits=4,
        decimal_places=2,
        widget=forms.NumberInput(
            attrs={
                'class': 'form-control',
                'step': '0.25',
                'min': '0',
                'max': '24',
                'placeholder': 'mis. 8',
                'inputmode': 'decimal',
            }
        ),
    )
    tariff_per_kwh = forms.DecimalField(
        label='Tarif listrik (Rp/kWh)',
        min_value=0,
        max_digits=12,
        decimal_places=2,
        widget=forms.NumberInput(
            attrs={
                'class': 'form-control',
                'step': '0.01',
                'placeholder': 'Masukkan tarif listrik',
                'inputmode': 'decimal',
            }
        ),
    )
