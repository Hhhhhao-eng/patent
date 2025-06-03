from django.urls import path
from .views import PatentRecommend, PatentDetail, KnowledgeGraphData, KGPCInference

app_name = 'kgpc'  # 添加这行很重要

urlpatterns = [
    path('patent_recommend/', PatentRecommend.as_view(), name='recommend'),
    path('patent/detail/', PatentDetail.as_view(), name='patent_detail'),
    path('graph/', KnowledgeGraphData.as_view(), name='knowledge_graph'),
    path('kgpc/infer/', KGPCInference.as_view(), name='kgpc_inference'),
    # 可以根据需要添加 path('kgpc/path/', ...) 等其他接口
]