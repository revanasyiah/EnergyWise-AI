from django.db import connection

from .models import EnergyHistoryRecord


def admin_overview(request):
    resolver_match = request.resolver_match
    if (
        not request.user.is_staff
        or resolver_match is None
        or resolver_match.namespace != 'admin'
        or resolver_match.url_name != 'index'
    ):
        return {}

    return {
        'energywise_database_engine': {
            'sqlite': 'SQLite',
            'postgresql': 'PostgreSQL',
            'mysql': 'MySQL',
            'oracle': 'Oracle',
        }.get(connection.vendor, connection.vendor),
        'energywise_history_count': EnergyHistoryRecord.objects.count(),
        'energywise_recent_history': EnergyHistoryRecord.objects.all()[:5],
    }
