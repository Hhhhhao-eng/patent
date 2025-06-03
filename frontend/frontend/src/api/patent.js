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
  }
}