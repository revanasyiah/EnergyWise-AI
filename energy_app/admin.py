from django.contrib import admin

from .models import EnergyHistoryRecord


@admin.register(EnergyHistoryRecord)
class EnergyHistoryRecordAdmin(admin.ModelAdmin):
    date_hierarchy = 'created_at'
    list_display = (
        'recorded_at',
        'record_type',
        'short_input_summary',
        'result_display',
    )
    list_filter = ('prediction_type', 'created_at')
    list_per_page = 50
    ordering = ('-created_at', '-pk')
    search_fields = ('input_summary', 'details')
    readonly_fields = ('created_at',)
    fieldsets = (
        (
            'Catatan energi',
            {
                'fields': (
                    'created_at',
                    'prediction_type',
                    'input_summary',
                    'result_value',
                    'result_unit',
                    'details',
                ),
            },
        ),
    )

    @admin.display(description='Ringkasan masukan')
    def short_input_summary(self, record):
        if len(record.input_summary) > 80:
            return f'{record.input_summary[:77]}...'
        return record.input_summary

    @admin.display(description='Waktu', ordering='created_at')
    def recorded_at(self, record):
        return record.created_at

    @admin.display(description='Jenis', ordering='prediction_type')
    def record_type(self, record):
        return record.get_prediction_type_display()

    @admin.display(description='Hasil')
    def result_display(self, record):
        return f'{record.result_value} {record.result_unit}'


admin.site.site_header = 'Administrasi EnergyWise AI'
admin.site.site_title = 'EnergyWise AI Admin'
admin.site.index_title = 'Pengelolaan data energi'
