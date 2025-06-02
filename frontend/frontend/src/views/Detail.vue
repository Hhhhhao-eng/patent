<template>
  <div class="detail-container">
    <el-page-header @back="$router.go(-1)" title="返回" />
    <el-card v-if="patent">
      <template #header>
        <h2>{{ patent.title }}</h2>
        <div class="patent-meta">
          <el-tag type="success">{{ patent.patType }}</el-tag>
          <span>申请号: {{ patent.appNumber }}</span>
          <span>申请日: {{ patent.appDate }}</span>
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
    async fetchPatentDetail() {
      try {
        this.loading = true;
        const response = await patentApi.getPatentDetail(this.id);
        this.patent = response.data;
      } catch (err) {
        this.$message.error('获取专利详情失败');
      } finally {
        this.loading = false;
      }
    },
    
    showGraph(patent) {
      this.$router.push({
        name: 'PatentGraph',
        params: { id: patent.id }
      });
    }
  }
}
</script>

<style scoped>
.detail-container {
  max-width: 1000px;
  margin: 20px auto;
  padding: 20px;
}

.patent-meta {
  margin-top: 10px;
  display: flex;
  align-items: center;
  gap: 15px;
  color: #666;
  font-size: 14px;
}

.patent-info {
  margin: 20px 0;
  line-height: 2;
}

.patent-abstract {
  margin-top: 30px;
  padding-top: 20px;
  border-top: 1px solid #eee;
}

.actions {
  margin-top: 30px;
  text-align: center;
}
</style>