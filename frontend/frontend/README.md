# patent

## Project setup
```
npm install
```

### Compiles and hot-reloads for development
```
npm run serve
```

### Compiles and minifies for production
```
npm run build
```

### Lints and fixes files
```
npm run lint
```

### Customize configuration
See [Configuration Reference](https://cli.vuejs.org/config/).

### 额外配置
前端：npm install -g @vue/cli 
npm install d3 echarts
npm install element-plus
npm install axios
后端：pip install django-cors-headers（在settings.py中添加corsheaders到INSTALLED_APPS，并配置CORS_ALLOW_ALL_ORIGINS=True）

### 还需要的后端接口
1. 专利详情接口
- 用于Detail.vue页面，获取某个专利的详细信息。
- 建议接口：/api/patent/detail?patent_id=xxx
2. 知识图谱数据接口
- 用于Graph.vue，返回某个专利或实体的知识图谱结构数据。
- 建议接口：/api/graph?patent_id=xxx 或 /api/graph/entity?id=xxx
3. AI/知识图谱推理相关接口
- 用于AI/KGPC/index.vue，如实体关系推理、路径发现等。
- 建议接口：如 /api/kgpc/infer、/api/kgpc/path