<template>
  <div class="kgpc-container">
    <h2 style="margin-bottom: 24px;">专利聚类推荐平台</h2>
    <el-input v-model="searchObj.patent_id" placeholder="输入专利号（pid）进行聚类推荐" style="width: 300px; margin-right: 10px;" />
    <el-button type="primary" @click="searchRecommend">聚类推荐</el-button>
    <el-alert v-if="error" :title="error" type="error" show-icon style="margin: 20px 0;" />
    <el-table v-if="list.length" :data="list" style="width: 100%; margin-top: 20px;">
      <el-table-column prop="title" label="专利名称" width="300" />
      <el-table-column prop="appNumber" label="申请号" width="180" />
      <el-table-column prop="appDate" label="申请日" width="120" />
      <el-table-column prop="applicantName" label="申请人" width="200" />
      <el-table-column label="操作">
        <template #default="{ row }">
          <el-button size="mini" @click="showDetail(row)">详情</el-button>
        </template>
      </el-table-column>
    </el-table>
    <el-empty v-else description="暂无推荐结果" style="margin-top: 40px;" />
  </div>
</template>

<script>
import patentApi from '@/api/patent';

export default {
  name: 'KGPC',
  data() {
    return {
      searchObj: {
        patent_id: ''
      },
      list: [],
      error: ''
    };
  },
  methods: {
    async searchRecommend() {
      this.error = '';
      this.list = [];
      if (!this.searchObj.patent_id) {
        this.error = '请输入专利号（pid）';
        return;
      }
      // 自动转为大写，避免大小写不一致导致查不到
      const patentId = this.searchObj.patent_id.trim().toUpperCase();
      try {
        const res = await patentApi.searchPatents({ patent_id: patentId });
        console.log('接口响应数据:', res.data); // 调试输出
        if (res.data && res.data.obj) {
          this.list = res.data.obj;
        } else if(res.data && res.data.error) {
          // 如果有可用专利号，给出友好提示
          if(res.data.available_pids) {
            this.error = res.data.error + '，可用专利号示例：' + res.data.available_pids.join('，');
          } else {
            this.error = res.data.error;
          }
        } else {
          this.error = '未获取到推荐结果';
        }
      } catch (e) {
        console.log('接口异常', e, e.response); // 调试输出
        // 处理404时的后端返回内容
        if(e.response && e.response.data && e.response.data.error && e.response.data.available_pids) {
          this.error = e.response.data.error + '，可用专利号示例：' + e.response.data.available_pids.join('，');
        } else {
          this.error = e.response?.data?.error || '推荐接口请求失败';
        }
      }
    },
    showDetail(row) {
      // 详情页跳转可根据后端接口完善
      // this.$router.push({ name: 'PatentDetail', params: { id: row.id } });
    }
  }
};
</script>

<style scoped>
.kgpc-container {
  max-width: 900px;
  margin: 40px auto;
  padding: 30px;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 8px #f0f1f2;
}
</style>
