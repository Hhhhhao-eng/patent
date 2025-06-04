<template>
  <div class="infer-container">
    <div class="card">
      <div class="header-row">
        <h2>关系推理 <span class="subtitle">（智能分析两个实体间可能的语义关系）</span></h2>
      </div>
      <el-form :inline="true" class="form-row" @submit.prevent="infer">
        <el-form-item label="实体1">
          <el-autocomplete v-model="entity1" :fetch-suggestions="querySearchEntity1" :trigger-on-focus="false" :debounce="200" :clearable="true" placeholder="专利标题" />
        </el-form-item>
        <el-form-item label="实体2">
          <el-autocomplete v-model="entity2" :fetch-suggestions="querySearchEntity2" :trigger-on-focus="false" :debounce="200" :clearable="true" placeholder="专利标题" />
        </el-form-item>
        <el-form-item label="Top K">
          <el-input-number v-model="topK" :min="1" :max="10" size="small" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="infer" :loading="loading">推理</el-button>
        </el-form-item>
      </el-form>
      <el-alert v-if="error" :title="error" type="error" show-icon style="margin:20px 0;" />
      <el-table v-if="results.length" :data="results" style="margin-top:20px;" stripe border size="small">
        <el-table-column prop="relation" label="关系" width="180" />
        <el-table-column prop="score" label="置信度" width="120">
          <template #default="scope">
            <el-tag type="success">{{ scope.row.score.toFixed(4) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="explanation" label="说明" />
      </el-table>
      <el-empty v-else-if="!loading && !results.length" description="暂无推理结果" style="margin-top:40px;" />
      <div v-if="loading" class="loading-bar">
        <el-icon><i class="el-icon-loading"></i></el-icon>
        <span>推理中，请稍候...</span>
      </div>
    </div>
  </div>
</template>

<script>
import patentApi from '@/api/patent';
export default {
  name: 'RelationInfer',
  data() {
    return {
      entity1: '',
      entity2: '',
      topK: 5,
      results: [],
      error: '',
      loading: false,
      entityOptions: [] // 实体名选项
    }
  },
  async mounted() {
    // 获取实体名列表用于自动补全
    try {
      const res = await patentApi.getEntityList();
      this.entityOptions = res.data.entities || [];
    } catch (e) {
      this.entityOptions = [];
    }
  },
  methods: {
    querySearchEntity1(query, cb) {
      const results = this.entityOptions.filter(e => e.includes(query));
      cb(results.map(e => ({ value: e })));
    },
    querySearchEntity2(query, cb) {
      const results = this.entityOptions.filter(e => e.includes(query));
      cb(results.map(e => ({ value: e })));
    },
    async infer() {
      this.error = '';
      this.results = [];
      if (!this.entity1 || !this.entity2) {
        this.error = '请输入两个实体名';
        return;
      }
      this.loading = true;
      try {
        const res = await patentApi.inferRelation(this.entity1, this.entity2, this.topK);
        if (res.data && res.data.results) {
          this.results = res.data.results;
        } else {
          this.error = res.data.error || '未获取到推理结果';
        }
      } catch (e) {
        this.error = e.response?.data?.error || '推理接口请求失败';
      } finally {
        this.loading = false;
      }
    }
  }
}
</script>

<style scoped>
.infer-container {
  max-width: 800px;
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
  align-items: baseline;
  margin-bottom: 18px;
}
.header-row h2 {
  font-size: 1.6rem;
  font-weight: 600;
  margin: 0;
}
.subtitle {
  font-size: 1rem;
  color: #888;
  margin-left: 16px;
  font-weight: 400;
}
.form-row {
  margin-bottom: 10px;
}
.loading-bar {
  display: flex;
  align-items: center;
  color: #409EFF;
  font-size: 1.1rem;
  margin: 30px 0 10px 0;
  gap: 8px;
}
</style>
