import axios from 'axios';

const apiClient = axios.create({
  baseURL: 'http://localhost:8000', // Django后端服务地址
  timeout: 10000
});

export default {
  // 专利聚类推荐（专利号为pid，接口为/api/patent_recommend/?patent_id=xxx）
  searchPatents(params) {
    return apiClient.get('/api/patent_recommend/', { params });
  },
  // 获取专利详情（接口为 /api/patent_detail/?patent_id=xxx）
  getPatentDetail(patent_id) {
    return apiClient.get('/api/patent_detail/', { params: { patent_id } });
  },
  // 获取知识图谱数据（接口为 /api/graph/?patent_id=xxx 或 /api/graph/?entity_id=xxx）
  getPatentGraph(patent_id) {
    return apiClient.get('/api/graph/', { params: { patent_id } });
  },
  // 导出知识图谱到Neo4j（接口为 /api/graph/neo4j/?patent_id=xxx）
  exportGraphToNeo4j(patent_id) {
    return apiClient.get('/api/graph/neo4j/', { params: { patent_id } });
  },
  // 实体关系推理（接口为 /api/kgpc/infer/relation/?entity1=xxx&entity2=xxx&top_k=5）
  inferRelation(entity1, entity2, top_k = 5) {
    return apiClient.get('/api/kgpc/infer/relation/', { params: { entity1, entity2, top_k } });
  },
  // 实体路径发现（接口为 /api/kgpc/infer/path/?entity1=xxx&entity2=xxx&max_length=3&top_k=5）
  findPath(entity1, entity2, max_length = 3, top_k = 5) {
    return apiClient.get('/api/kgpc/infer/path/', { params: { entity1, entity2, max_length, top_k } });
  },
  // 获取实体名列表（接口为 /api/kgpc/entity_list/）
  getEntityList() {
    return apiClient.get('/api/kgpc/entity_list/');
  },
}