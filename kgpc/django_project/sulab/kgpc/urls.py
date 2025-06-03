from django.urls import path
from .views import PatentRecommend, PatentDetail, KnowledgeGraphData, Neo4jExport,KGPCInferRelation,KGPCFindPath

app_name = 'kgpc'  # 添加这行很重要

urlpatterns = [
    path('patent_recommend/', PatentRecommend.as_view(), name='recommend'),
    path('patent/detail/', PatentDetail.as_view(), name='patent_detail'),
    path('graph/', KnowledgeGraphData.as_view(), name='knowledge_graph'),
    path('graph/neo4j/', Neo4jExport.as_view(), name='knowledge_graph_neo4j'),
    path('kgpc/infer/relation/', KGPCInferRelation.as_view(), name='kgpc_infer_relation'),
    path('kgpc/infer/path/', KGPCFindPath.as_view(), name='kgpc_find_path'),
    # 可以根据需要添加 path('kgpc/path/', ...) 等其他接口
]