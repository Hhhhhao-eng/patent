# Create your models here.

import torch
from .utils.entity_clustering import KnowledgeCompletion, load_refined_embeddings, load_triplets_from_files


class KnowledgeCompletionModel:
    def __init__(self):
        self.model = None
        self.entity2id = None
        self.id2entity = None

    def load(self, embedding_path, triplet_files):
        # 加载嵌入和三元组
        refined_embeddings = load_refined_embeddings(embedding_path)
        triplets = load_triplets_from_files(triplet_files)

        # 初始化模型
        self.model = KnowledgeCompletion(
            entity_embeddings=refined_embeddings,
            triplets=triplets,
            num_relations=4,
            embedding_dim=len(next(iter(refined_embeddings.values()))),
            num_clusters=5
        )

        # 加载预训练权重
        self.model.load_state_dict(torch.load('best_sacn_model.pth'))

        # 保存映射关系
        self.entity2id = self.model.entity2id
        self.id2entity = self.model.id2entity

    def cluster_entities(self, use_kmeans=True):
        return self.model.cluster_entities(self.model, use_kmeans=use_kmeans)

    def predict_missing_links(self, head_entity, relation, top_k=10):
        return self.model.predict_missing_links(self.model, head_entity, relation, top_k)

    def visualize_clusters(self, entity_clusters, cluster_centers=None, **kwargs):
        return self.model.visualize_clusters(self.model, entity_clusters, cluster_centers, **kwargs)
