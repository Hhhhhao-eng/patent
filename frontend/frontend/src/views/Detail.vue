<template>
  <div class="detail-container">
    <div class="card">
      <el-page-header @back="goBackWithState" title="返回" />
      <el-card v-if="patent" shadow="hover">
        <template #header>
          <div class="header-row">
            <h2>{{ patent.title }}</h2>
            <div class="patent-meta">
              <el-tag type="success">{{ patent.patType }}</el-tag>
              <span>申请号: {{ patent.appNumber }}</span>
              <span>申请日: {{ patent.appDate }}</span>
            </div>
          </div>
        </template>
        <div class="patent-info">
          <div><strong>申请人:</strong> {{ patent.applicantName }}</div>
          <div><strong>发明人:</strong> {{ patent.inventorName }}</div>
          <div><strong>分类号:</strong> {{ patent.mainIpc }} ({{ patent.ipc }})</div>
          <div><strong>地址:</strong> {{ patent.address }}</div>
        </div>
        <div class="patent-abstract">
          <h3>摘要</h3>
          <p>{{ patent.abs }}</p>
        </div>
        <div class="actions">
          <el-button type="primary" @click="showGraph(patent)">查看知识图谱</el-button>
        </div>
      </el-card>
      <el-skeleton v-else :rows="5" animated />
    </div>
  </div>
</template>

<script>
import patentApi from '@/api/patent';

export default {
  name: 'PatentDetail',
  props: ['id'],
  data() {
    return {
      patent: null,
      loading: false
    }
  },
  mounted() {
    this.fetchPatentDetail();
  },
  methods: {
    goBackWithState() {
      // 优先用路由 query 里的 patent_id，保证返回推荐页时恢复原始输入
      const backPatentId = this.$route.query.patent_id || (this.patent ? this.patent.pid : '');
      this.$router.push({
        name: 'KGPC',
        query: { patent_id: backPatentId }
      });
    },
    async fetchPatentDetail() {
      try {
        this.loading = true;
        // 兼容路由参数传递方式
        const patentId = this.id || this.$route.params.id;
        const response = await patentApi.getPatentDetail(patentId);
        // 适配后端返回结构
        this.patent = response.data.data;
      } catch (err) {
        this.$message.error('获取专利详情失败');
      } finally {
        this.loading = false;
      }
    },
    
    showGraph(patent) {
      // 跳转到知识图谱页，传递专利号
      this.$router.push({
        name: 'PatentGraph',
        params: { id: patent.pid }
      });
    }
  }
}
</script>

<style scoped>
.detail-container {
  max-width: 1000px;
  margin: 40px auto;
  padding: 0;
  background: #f6f8fa;
}
.card {
  background: #fff;
  border-radius: 10px;
  box-shadow: 0 2px 12px #e6e6e6;
  padding: 32px 36px 28px 36px;
}
.header-row {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.header-row h2 {
  font-size: 1.5rem;
  font-weight: 600;
  margin: 0;
}
.patent-meta {
  display: flex;
  align-items: center;
  gap: 18px;
  color: #666;
  font-size: 14px;
}
.patent-info {
  margin: 20px 0;
  line-height: 2;
  font-size: 1.08rem;
}
.patent-abstract {
  margin-top: 30px;
  padding-top: 20px;
  border-top: 1px solid #eee;
}
.patent-abstract h3 {
  font-size: 1.1rem;
  color: #409EFF;
  margin-bottom: 8px;
}
.actions {
  margin-top: 30px;
  text-align: center;
}
</style>