<template>
  <div class="kgpc-container">
    <div class="card">
      <div class="header-row">
        <h2>专利聚类推荐平台 <span class="subtitle">（智能推荐相似专利，支持知识图谱分析）</span></h2>
      </div>
      <div class="nav-btns">
        <el-button @click="$router.push({name:'RelationInfer'})">关系推理</el-button>
        <el-button @click="$router.push({name:'PathFinder'})">路径发现</el-button>
      </div>
      <div class="search-row">
        <el-input v-model="searchObj.patent_id" placeholder="输入专利号（pid）进行聚类推荐" clearable style="width: 300px; margin-right: 10px;" />
        <el-autocomplete
          v-model="searchObj.patent_title"
          :fetch-suggestions="querySearchTitle"
          placeholder="或输入专利名称自动补全"
          clearable
          style="width: 300px; margin-right: 10px;"
          @select="handleTitleSelect"
        />
        <el-button type="primary" @click="searchRecommend">聚类推荐</el-button>
      </div>
      <el-alert v-if="error" :title="error" type="error" show-icon style="margin: 20px 0;" />
      <el-table v-if="list.length" :data="list" style="width: 100%; margin-top: 20px;" stripe border size="small">
        <el-table-column prop="title" label="专利名称" width="300" />
        <el-table-column prop="appNumber" label="申请号" width="180" />
        <el-table-column prop="appDate" label="申请日" width="120" />
        <el-table-column prop="applicantName" label="申请人" width="200" />
        <el-table-column label="操作" width="120">
          <template #default="{ row }">
            <el-button size="mini" @click="showDetail(row)">详情</el-button>
          </template>
        </el-table-column>
      </el-table>
      <el-empty v-else description="暂无推荐结果" style="margin-top: 40px;" />
    </div>
  </div>
</template>

<script>
import patentApi from '@/api/patent';

export default {
  name: 'KGPC',
  data() {
    return {
      searchObj: {
        patent_id: '',
        patent_title: ''
      },
      list: [],
      error: '',
      entityList: []
    };
  },
  mounted() {
    // 如果通过 query 传递了 patent_id，则自动查找推荐
    const queryId = this.$route.query.patent_id;
    if (queryId) {
      this.searchObj.patent_id = queryId;
      this.searchRecommend();
    }
    this.getEntityList();
  },
  methods: {
    async getEntityList() {
      // 获取实体名列表（专利名）
      try {
        const res = await patentApi.getEntityList();
        if (res.data && res.data.entities) {
          this.entityList = res.data.entities;
        }
      } catch (e) {
        // 忽略异常
      }
    },
    querySearchTitle(queryString, cb) {
      // 支持拼音/模糊/部分匹配，提升补全体验
      if (!queryString) {
        cb([]);
        return;
      }
      // 支持首字母、全拼、部分匹配
      const lowerQuery = queryString.toLowerCase();
      const results = this.entityList.filter(name => {
        return (
          name.includes(queryString) ||
          name.toLowerCase().includes(lowerQuery) ||
          this.getPinyin(name).includes(lowerQuery)
        );
      });
      cb(results.map(name => ({ value: name })));
    },
    getPinyin(str) {
      // 简单拼音首字母匹配（可用第三方库优化）
      return str
        .split('')
        .map(char => {
          // 汉字转拼音首字母，非汉字原样返回
          const code = char.charCodeAt(0);
          if (code >= 19968 && code <= 40869) {
            // 简单映射，建议用 pinyin 库
            return window.PINYIN_FIRST_LETTER ? window.PINYIN_FIRST_LETTER[char] || '' : '';
          }
          return char;
        })
        .join('');
    },
    handleTitleSelect(item) {
      // 选中专利名后，自动查找对应pid
      this.searchObj.patent_title = item.value;
      this.findPidByTitle(item.value);
    },
    async findPidByTitle(title) {
      try {
        const response = await fetch('/pid_title_map.txt');
        if (response.ok) {
          const text = await response.text();
          const lines = text.split('\n');
          const normalize = str => str.replace(/\s|\p{P}|\u3000/gu, '').toLowerCase();
          const getPinyin = this.getPinyin;
          const input = normalize(title);
          let strictPid = '';
          let fuzzyPid = '';
          for (const line of lines) {
            if (!line) continue;
            const [pid, t] = line.split('\t');
            if (!t) continue;
            const tNorm = normalize(t);
            // 严格匹配（全等或拼音全等）
            if (
              tNorm === input ||
              (getPinyin && getPinyin(t).toLowerCase() === getPinyin(title).toLowerCase())
            ) {
              strictPid = pid;
              break;
            }
          }
          if (strictPid) {
            this.searchObj.patent_id = strictPid;
            this.searchRecommend();
            return;
          }
          // 模糊匹配（包含、去空格、大小写不敏感等）
          for (const line of lines) {
            if (!line) continue;
            const [pid, t] = line.split('\t');
            if (!t) continue;
            const tNorm = normalize(t);
            if (
              tNorm.includes(input) ||
              (getPinyin && getPinyin(t).toLowerCase().includes(getPinyin(title).toLowerCase()))
            ) {
              if (pid !== 'CN100000') {
                fuzzyPid = pid;
                break;
              }
            }
          }
          this.searchObj.patent_id = fuzzyPid;
          this.searchRecommend();
        } else {
          this.searchObj.patent_id = '';
        }
      } catch (e) {
        this.searchObj.patent_id = '';
      }
    },
    async searchRecommend() {
      this.error = '';
      this.list = [];
      if (!this.searchObj.patent_id) {
        if (this.searchObj.patent_title) {
          // 用户输入了专利名但未选中，尝试查找pid
          await this.findPidByTitle(this.searchObj.patent_title);
          if (!this.searchObj.patent_id) {
            this.error = '未找到对应专利号，请检查专利名称或选择自动补全项';
            return;
          }
        } else {
          this.error = '请输入专利号（pid）或选择专利名称';
          return;
        }
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
      // 跳转到详情页，带上当前输入的 patent_id 作为 query，便于返回时恢复
      this.$router.push({ name: 'PatentDetail', params: { id: row.pid }, query: { patent_id: this.searchObj.patent_id } });
    },
    showGraph(row) {
      // 跳转到知识图谱页，传递专利号
      this.$router.push({ name: 'PatentGraph', params: { id: row.pid } });
    }
  }
};
</script>

<style scoped>
.kgpc-container {
  max-width: 950px;
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
.nav-btns {
  margin-bottom: 18px;
}
.search-row {
  display: flex;
  align-items: center;
  margin-bottom: 10px;
}
</style>
