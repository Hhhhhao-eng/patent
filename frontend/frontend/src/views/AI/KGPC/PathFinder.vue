<template>
  <div class="pathfinder-container">
    <div class="card">
      <div class="header-row">
        <h2>路径发现 <span class="subtitle">（智能发现两个实体间的多跳语义路径）</span></h2>
      </div>
      <el-form :inline="true" class="form-row" @submit.prevent="findPath">
        <el-form-item label="实体1">
          <el-autocomplete v-model="entity1" :fetch-suggestions="querySearchEntity1" :trigger-on-focus="false" :debounce="200" :clearable="true" placeholder="专利标题" />
        </el-form-item>
        <el-form-item label="实体2">
          <el-autocomplete v-model="entity2" :fetch-suggestions="querySearchEntity2" :trigger-on-focus="false" :debounce="200" :clearable="true" placeholder="专利标题" />
        </el-form-item>
        <el-form-item label="最大路径长度">
          <el-input-number v-model="maxLength" :min="2" :max="5" size="small" />
        </el-form-item>
        <el-form-item label="Top K">
          <el-input-number v-model="topK" :min="1" :max="10" size="small" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="findPath" :loading="loading">发现路径</el-button>
        </el-form-item>
      </el-form>
      <el-alert v-if="error" :title="error" type="error" show-icon style="margin:20px 0;" />
      <el-table v-if="paths.length" :data="paths" style="margin-top:20px;" stripe border size="small">
        <el-table-column label="路径详情">
          <template #default="{ row }">
            <div class="path-row">
              <span v-for="(step, idx) in row" :key="idx">
                <span class="entity">{{ step.source }}</span>
                <span class="arrow">--{{ step.relation }}→</span>
                <span v-if="idx === row.length-1" class="entity">{{ step.target }}</span>
              </span>
            </div>
          </template>
        </el-table-column>
      </el-table>
      <el-empty v-else-if="!loading && !paths.length" description="暂无路径结果" style="margin-top:40px;" />
      <div v-if="loading" class="loading-bar">
        <el-icon><i class="el-icon-loading"></i></el-icon>
        <span>路径推理中，请稍候...</span>
      </div>
    </div>
  </div>
</template>

<script>
import patentApi from '@/api/patent';
export default {
  name: 'PathFinder',
  data() {
    return {
      entity1: '',
      entity2: '',
      maxLength: 3,
      topK: 5,
      paths: [],
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
    async findPath() {
      this.error = '';
      this.paths = [];
      if (!this.entity1 || !this.entity2) {
        this.error = '请输入两个实体名';
        return;
      }
      this.loading = true;
      try {
        const res = await patentApi.findPath(this.entity1, this.entity2, this.maxLength, this.topK);
        if (res.data && res.data.paths) {
          this.paths = res.data.paths;
        } else {
          this.error = res.data.error || '未获取到路径结果';
        }
      } catch (e) {
        this.error = e.response?.data?.error || '路径接口请求失败';
      } finally {
        this.loading = false;
      }
    }
  }
}
</script>

<style scoped>
.pathfinder-container {
  max-width: 900px;
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
.path-row {
  font-size: 1.08rem;
  line-height: 2.1;
  word-break: break-all;
}
.entity {
  color: #222;
  font-weight: 500;
}
.arrow {
  color: #409EFF;
  margin: 0 4px;
}
</style>
