import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class RelationGenerator:
    def __init__(self, knowledge_completion):
        self.kc = knowledge_completion

    def generate_similar_to_relations(self, model, output_file, cluster_threshold=0.8, intra_cluster_only=True):
        """
        生成similar_to关系并保存到文件

        Args:
            model: 训练好的SACN模型
            output_file: 输出文件路径
            cluster_threshold: 相似度阈值(0-1)
            intra_cluster_only: 是否只添加同一聚类内的相似关系
        """
        # 获取聚类结果
        entity_clusters, _ = self.kc.cluster_entities(model, use_kmeans=True)

        # 获取所有实体嵌入
        with torch.no_grad():
            all_entities = torch.arange(self.kc.num_entities).to(self.kc.device)
            entity_embeddings = model.emb_e(all_entities).cpu().numpy()

        # 计算相似度矩阵
        similarity_matrix = cosine_similarity(entity_embeddings)

        # 生成关系
        relations = self._generate_relations(
            entity_clusters,
            similarity_matrix,
            cluster_threshold,
            intra_cluster_only
        )

        # 保存到文件
        self._save_relations(relations, output_file)

        return relations

    def _generate_relations(self, entity_clusters, similarity_matrix, threshold, intra_cluster_only):
        """生成相似关系对"""
        relations = set()

        for i in range(self.kc.num_entities):
            entity_i = self.kc.id2entity[i]
            cluster_i = entity_clusters.get(entity_i, -1)

            for j in range(i + 1, self.kc.num_entities):
                entity_j = self.kc.id2entity[j]
                cluster_j = entity_clusters.get(entity_j, -1)

                if intra_cluster_only and cluster_i != cluster_j:
                    continue

                sim_score = similarity_matrix[i][j]
                if sim_score >= threshold:
                    relations.add((
                        min(entity_i, entity_j),
                        "similar_to",
                        max(entity_i, entity_j),
                        float(sim_score)
                    ))

        return relations

    def _save_relations(self, relations, output_file):
        """保存关系到文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for rel in sorted(relations, key=lambda x: -x[3]):  # 按相似度降序排序
                f.write(f"{rel[0]}---{rel[1]}---{rel[2]}---{rel[3]:.4f}\n")

        print(f"Saved {len(relations)} similar_to relations to {output_file}")