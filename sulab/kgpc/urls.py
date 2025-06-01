from django.urls import path
from .views import PatentRecommend

app_name = 'kgpc'  # 添加这行很重要

urlpatterns = [
    path('patent_recommend/', PatentRecommend.as_view(), name='recommend'),
]