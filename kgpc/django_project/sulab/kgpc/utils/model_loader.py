import os
import torch
import numpy as np
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

# 全局资源变量
_initialized = False
kc_model = None
sacn_model = None
entity2id = None
id2entity = None
patent_data_dict = {}
recommend_relation = "similar_to"


def load_resources():
    """加载所有必要资源"""
    global _initialized, kc_model, sacn_model, entity2id, id2entity, patent_data_dict

    if _initialized:
        return

    try:
        logger.info("Starting resource loading...")

        # 1. 检查文件是否存在
        required_files = [
            settings.EMBEDDINGS_PATH,
            settings.TRIPLETS_PATH,
            settings.MODEL_PATH
        ]

        for filepath in required_files:
            if not os.path.exists(filepath):
                logger.error(f"Required file not found: {filepath}")
                raise FileNotFoundError(f"Required file not found: {filepath}")

        # 2. 加载实体嵌入
        from .entity_clustering import load_refined_embeddings, load_triplets_from_files
        refined_embeddings = load_refined_embeddings(settings.EMBEDDINGS_PATH)
        logger.info(f"Loaded {len(refined_embeddings)} embeddings")

        # 3. 加载三元组
        # 加载包含similar_to关系的三元组
        files = [
            os.path.join(settings.BASE_DIR, 'data', 'triplets', 'new_title_triplets.txt'),
            os.path.join(settings.BASE_DIR, 'data', 'triplets', 'similar_to_relations.txt')
        ]
        triplets = load_triplets_from_files(files)
        logger.info(f"Loaded {len(triplets)} triplets")

        # 4. 初始化知识补全模块
        from .entity_clustering import KnowledgeCompletion, SACN
        # 初始化知识补全模块 - 使用5种关系
        kc = KnowledgeCompletion(
            entity_embeddings=refined_embeddings,
            triplets=triplets,
            num_relations=5,  # 包含similar_to
            embedding_dim=len(next(iter(refined_embeddings.values()))),
            num_clusters=10
        )

        # 创建模型时也使用5种关系
        model = SACN(
            num_entities=kc.num_entities,
            num_relations=kc.num_relations,  # 这会自动使用5
            embedding_dim=kc.embedding_dim,
            init_embeddings=kc.entity_emb_matrix,
            num_clusters=10
        )
        model.load_state_dict(torch.load(settings.MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()

        # 6. 构建专利数据
        build_sample_patent_data()

        # 7. 设置全局变量
        kc_model = kc
        sacn_model = model
        entity2id = kc.entity2id
        id2entity = kc.id2entity
        _initialized = True

        logger.info(f"Initialization completed. Loaded {len(patent_data_dict)} patents")

    except Exception as e:
        logger.error(f"Resource loading failed: {str(e)}", exc_info=True)
        raise


def get_resources():
    """获取已加载的资源"""
    if not _initialized:
        load_resources()  # 尝试重新初始化

    return {
        "initialized": _initialized,
        "kc_model": kc_model,
        "sacn_model": sacn_model,
        "entity2id": entity2id,
        "id2entity": id2entity,
        "patent_data_dict": patent_data_dict,
        "recommend_relation": recommend_relation
    }


def build_sample_patent_data():
    """从三元组文件中构建专利数据字典，使用pid作为主键"""
    global patent_data_dict, patent_title_to_pid

    patent_data_dict = {}
    patent_title_to_pid = {}  # 新增：标题到专利号的映射
    triplets_file = os.path.join(settings.BASE_DIR, 'data', 'triplets', 'new_title_triplets.txt')

    if not os.path.exists(triplets_file):
        print(f"Warning: Triplets file {triplets_file} not found")
        return

    current_patent = None
    with open(triplets_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.count('---') < 2:
                continue

            parts = line.split('---')
            patent_title = parts[0].strip()
            relation_type = parts[1].strip()
            values_part = parts[2].strip() if len(parts) > 2 else ''

            # 生成唯一专利号（如果尚未存在）
            if patent_title not in patent_title_to_pid:
                patent_number = f"CN{len(patent_title_to_pid) + 100000}"
                patent_title_to_pid[patent_title] = patent_number

            pid = patent_title_to_pid[patent_title]

            # 初始化专利数据结构（使用pid作为键）
            if pid not in patent_data_dict:
                patent_data_dict[pid] = [
                    pid,  # 专利号
                    patent_title,  # 专利标题
                    '发明',  # 专利类型
                    f"2020{len(patent_title_to_pid) + 10000}",  # 申请号
                    "2020-01-01",  # 申请日
                    f"{pid}A",  # 公开号
                    "2022-01-01",  # 授权日
                    "H01L",  # 主分类号
                    "",  # 分类号
                    "",  # 申请人
                    "",  # 发明人
                    "中国",  # 地址
                    patent_title,  # 摘要
                    "",  # 关键词
                    "",  # 背景技术
                    ""  # 权利要求
                ]

            # 获取当前专利数据引用
            current_patent = patent_data_dict[pid]

            # 处理关系数据
            values = []
            if values_part:
                if values_part.startswith('[') and values_part.endswith(']'):
                    try:
                        values = eval(values_part)
                        if not isinstance(values, list):
                            values = [str(values)]
                    except:
                        values = [values_part.strip("'\"")]
                else:
                    values = [values_part.strip("'\"")]

            # 更新专利数据（累加而不是覆盖）
            if relation_type == 'titleKey' and values:
                if current_patent[13]:
                    current_patent[13] += ';' + ';'.join(values)
                else:
                    current_patent[13] = ';'.join(values)
            elif relation_type == 'clKey' and values:
                if current_patent[15]:
                    current_patent[15] += ';' + ';'.join(values)
                else:
                    current_patent[15] = ';'.join(values)
            elif relation_type == 'bgKey' and values:
                if current_patent[14]:
                    current_patent[14] += ';' + ';'.join(values)
                else:
                    current_patent[14] = ';'.join(values)
            elif relation_type == 'patentee' and values:
                if current_patent[9]:
                    current_patent[9] += ';' + ';'.join(values)
                else:
                    current_patent[9] = ';'.join(values)

    print(f"Built patent data for {len(patent_data_dict)} patents (using PID as key)")


if __name__ == "__main__":
    # 自动导出 pid-title 对应表
    from kgpc.utils.model_loader import get_resources
    resources = get_resources()
    patent_data_dict = resources.get("patent_data_dict", {})
    with open("pid_title_map.txt", "w", encoding="utf-8") as f:
        for pid, data in patent_data_dict.items():
            f.write(f"{pid}\t{data[1]}\n")
    print("已导出 pid_title_map.txt")