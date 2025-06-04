<template>
  <div class="container">
    <!-- 搜索区域 -->
    <div class="search-section">
      <el-input 
        v-model="searchQuery" 
        placeholder="输入专利号、关键词或技术领域" 
        @keyup.enter="searchPatents"
      >
        <template #append>
          <el-button icon="el-icon-search" @click="searchPatents" />
        </template>
      </el-input>
      
      <div class="filters">
        <el-select v-model="filters.type" placeholder="专利类型">
          <el-option label="全部" value="" />
          <el-option label="发明专利" value="发明" />
          <el-option label="实用新型" value="实用新型" />
        </el-select>
        
        <el-date-picker
          v-model="filters.dateRange"
          type="daterange"
          range-separator="至"
          start-placeholder="申请开始日"
          end-placeholder="申请结束日"
        />
      </div>
    </div>

    <!-- 结果展示 -->
    <div v-loading="loading">
      <el-alert v-if="error" :title="error" type="error" show-icon />
      
      <div v-if="results.length > 0">
        <div class="result-summary">找到 {{ total }} 条相关专利</div>
        
        <el-table :data="results" style="width: 100%">
          <el-table-column prop="title" label="专利名称" width="300" />
          <el-table-column prop="appNumber" label="申请号" width="180" />
          <el-table-column prop="appDate" label="申请日" width="120" />
          <el-table-column prop="applicantName" label="申请人" width="200" />
          <el-table-column label="操作">
            <template #default="{ row }">
              <el-button size="mini" @click="showDetail(row)">详情</el-button>
              <el-button size="mini" type="primary" @click="showGraph(row)">知识图谱</el-button>
            </template>
          </el-table-column>
        </el-table>
        
        <el-pagination
          background
          layout="prev, pager, next"
          :total="total"
          :page-size="pageSize"
          @current-change="handlePageChange"
        />
      </div>
      
      <el-empty v-else description="暂无数据" />
    </div>
  </div>
</template>

<script>
import patentApi from '@/api/patent';

export default {
  data() {
    return {
      searchQuery: '',
      loading: false,
      error: '',
      results: [],
      total: 0,
      page: 1,
      pageSize: 10,
      filters: {
        type: '',
        dateRange: []
      }
    }
  },
  methods: {
    async searchPatents() {
      try {
        this.loading = true;
        this.error = '';
        // 兼容聚类推荐API
        const params = {
          patent_id: this.searchQuery || '',
          // 其他筛选条件可扩展
        };
        const response = await patentApi.searchPatents(params);
        this.results = response.data.obj || [];
        this.total = this.results.length;
      } catch (err) {
        this.error = err.response?.data?.message || '搜索失败';
      } finally {
        this.loading = false;
      }
    },
    
    showDetail(patent) {
      this.$router.push({
        name: 'PatentDetail',
        params: { id: patent.id }
      });
    },
    
    showGraph(patent) {
      this.$router.push({
        name: 'PatentGraph',
        params: { id: patent.id }
      });
    },
    
    handlePageChange(page) {
      this.page = page;
      this.searchPatents();
    }
  }
}
</script>

<style scoped>
.container {
  max-width: 1200px;
  margin: 20px auto;
  padding: 20px;
}

.search-section {
  margin-bottom: 30px;
}

.filters {
  margin-top: 15px;
  display: flex;
  gap: 15px;
}

.result-summary {
  margin: 15px 0;
  font-size: 14px;
  color: #666;
}

.el-pagination {
  margin-top: 20px;
  justify-content: center;
}
</style>