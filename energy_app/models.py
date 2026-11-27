from django.db import models


class EnergyHistoryRecord(models.Model):
    class RecordType(models.TextChoices):
        ESTIMATION = 'estimation', 'Estimasi'
        PREDICTION = 'prediction', 'Prediksi'

    created_at = models.DateTimeField(auto_now_add=True)
    prediction_type = models.CharField(
        max_length=16,
        choices=RecordType.choices,
    )
    input_summary = models.CharField(max_length=500)
    result_value = models.DecimalField(max_digits=15, decimal_places=3)
    result_unit = models.CharField(max_length=16)
    details = models.TextField(blank=True)

    class Meta:
        ordering = ('-created_at', '-pk')
        verbose_name = 'Riwayat energi'
        verbose_name_plural = 'Riwayat energi'

    def __str__(self):
        return (
            f'{self.get_prediction_type_display()} · '
            f'{self.created_at:%Y-%m-%d %H:%M}'
        )
