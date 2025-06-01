带有详细注释的Django项目结构说明：

```
sulab/
│
├── data/                      # 数据存储目录（非Django标准目录，自定义）
│   ├── embeddings/            # 实体嵌入向量数据
│   │   └── refined_patent_embeddings.txt  # 强化后的专利嵌入向量
│   ├── model/                 # 模型存储
│   │   ├── best_sacn_model.pth       # 原始SACN模型参数
│   │   └── best_sacn_model_v2.pth    # 更新后的模型参数
│   └── triplets/              # 三元组数据文件
│       ├── new_title_triplets.txt     # 原始三元组数据
│       └── similar_to_relations.txt   # 生成的相似关系数据
│
├── kgpc/                      # Django应用目录（专利知识图谱核心功能）
│   ├── migrations/            # 数据库迁移文件（自动生成）
│   ├── utils/                 # 工具模块
│   │   ├── __init__.py        # 包初始化文件
│   │   ├── entity_clustering.py  # 实体聚类算法实现
│   │   ├── model_loader.py    # 模型加载工具
│   │   └── relation_generator.py # 关系生成器
│   ├── __init__.py            # 应用包声明
│   ├── admin.py               # 后台管理配置
│   ├── apps.py                # 应用配置类
│   ├── models.py              # 数据模型定义
│   ├── tests.py               # 测试用例
│   ├── urls.py                # 应用级URL路由
│   └── views.py               # 视图处理逻辑
│
└── sulab/                     # 项目配置目录（与项目同名）
    ├── __init__.py            # Python包声明
    ├── asgi.py                # ASGI服务器配置
    ├── settings.py            # 项目全局设置
    ├── urls.py                # 项目级URL路由
    ├── wsgi.py                # WSGI服务器配置
    ├── check_urls.py          # URL检查工具（自定义）
    ├── db.sqlite3             # SQLite数据库文件（开发用）
    ├── manage.py              # Django命令行工具
    ├── README.md              # 项目说明文档
    ├── test_urls.py           # URL测试脚本（自定义）
    └── .gitignore             # Git忽略规则
```

