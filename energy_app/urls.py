from django.urls import path

from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('estimation/', views.estimation, name='estimation'),
    path('prediction/', views.prediction, name='prediction'),
    path('cost/', views.cost, name='cost'),
    path('history/', views.history, name='history'),
    path('history/<int:pk>/', views.history_detail, name='history_detail'),
]
