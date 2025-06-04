import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from collections import defaultdict
from torch.nn import Parameter
import random
import os
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment

class SACN(nn.Module):
    """基于强化实体表征的知识图谱补全模型（添加聚类功能）"""

    def __init__(self, num_entities, num_relations, embedding_dim, init_embeddings,
                 num_clusters=10,  # 新增聚类数量参数
                 input_dropout=0.1, dropout_rate=0.2, channels=32, kernel_size=3):
        super(SACN, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_clusters = num_clusters  # 保存聚类数量

        # 使用预训练的实体嵌入
        self.emb_e = nn.Embedding.from_pretrained(torch.FloatTensor(init_embeddings), freeze=False)

        # 关系嵌入
        self.emb_rel = nn.Embedding(num_relations, embedding_dim)

        # Dropout层
        self.inp_drop = nn.Dropout(input_dropout)
        self.hidden_drop = nn.Dropout(dropout_rate)
        self.feature_map_drop = nn.Dropout(dropout_rate)

        # 损失函数
        self.loss = nn.BCELoss()

        # 卷积层
        self.conv1 = nn.Conv1d(2, channels, kernel_size, stride=1, padding=int(math.floor(kernel_size / 2)))

        # 批归一化层
        self.bn0 = nn.BatchNorm1d(2)  # 输入通道数=2
        self.bn1 = nn.BatchNorm1d(channels)  # 卷积输出通道数
        self.bn2 = nn.BatchNorm1d(embedding_dim)  # 全连接层输出维度

        # 偏置项
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))

        # 全连接层
        self.fc = nn.Linear(embedding_dim * channels, embedding_dim)

        # 聚类层
        self.cluster_fc = nn.Linear(embedding_dim, num_clusters)  # 聚类全连接层

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        """初始化权重"""
        nn.init.xavier_normal_(self.emb_rel.weight.data)
        nn.init.xavier_normal_(self.conv1.weight.data)
        nn.init.xavier_normal_(self.fc.weight.data)
        nn.init.xavier_normal_(self.cluster_fc.weight.data)  # 初始化聚类层

    def kl_loss(self, q):
        """
        计算KL散度损失
        Args:
            q: 聚类分配软分布 (batch_size, num_clusters)
        Returns:
            kl_loss: KL散度损失值
        """
        # 计算目标分布P
        weight = q ** 2 / q.sum(0)  # 计算权重
        p = (weight.t() / weight.sum(1)).t()  # 归一化得到目标分布

        # 计算KL散度 (KL(P||Q))
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        return kl_loss

    def forward(self, e1, rel, return_cluster=False):
        """
        前向传播（添加聚类输出）
        Args:
            return_cluster: 是否返回聚类分布
        """
        batch_size = e1.shape[0]

        # 获取实体和关系嵌入
        e1_embedded = self.emb_e(e1)  # (batch_size, embedding_dim)
        rel_embedded = self.emb_rel(rel)  # (batch_size, embedding_dim)

        # 拼接实体和关系嵌入
        stacked_inputs = torch.stack([e1_embedded, rel_embedded], dim=1)  # (batch_size, 2, embedding_dim)

        # 批归一化
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)

        # 卷积层
        x = self.conv1(x)  # (batch_size, channels, embedding_dim)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)

        # 展平
        x = x.view(batch_size, -1)  # (batch_size, channels * embedding_dim)

        # 全连接层
        x = self.fc(x)  # (batch_size, embedding_dim)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        # 计算聚类分布
        cluster_logits = self.cluster_fc(x)  # (batch_size, num_clusters)
        cluster_q = F.softmax(cluster_logits, dim=1)

        # 计算所有实体的分数
        all_entities = self.emb_e.weight  # (num_entities, embedding_dim)
        kc_logits = torch.mm(x, all_entities.transpose(1, 0))  # (batch_size, num_entities)
        kc_logits += self.b.expand_as(kc_logits)  # 添加偏置

        # Sigmoid激活得到概率
        kc_pred = torch.sigmoid(kc_logits)

        if return_cluster:
            return kc_pred, cluster_q
        return kc_pred


class KnowledgeCompletion:
    """知识图谱知识补全模块（添加自监督聚类）"""

    def __init__(self, entity_embeddings, triplets, num_relations, embedding_dim=32,
                 num_clusters=10,  # 新增聚类数量参数
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters  # 保存聚类数量

        # 创建实体到索引的映射
        self.entity2id = {entity: idx for idx, entity in enumerate(entity_embeddings.keys())}
        self.id2entity = {idx: entity for entity, idx in self.entity2id.items()}
        self.num_entities = len(self.entity2id)
        self.patent_ids = {f"CN{idx+100000}" for entity, idx in self.entity2id.items()}
        self.entity_ids = {entity for entity, idx in self.entity2id.items()}

        # 确保relation2id包含similar_to
        self.relation2id = {
            'titleKey': 0,
            'clKey': 1,
            'bgKey': 2,
            'patentee': 3,
            'similar_to': 4  # 明确添加
        }
        self.id2relation = {v: k for k, v in self.relation2id.items()}

        # 重新计算实际关系数量
        self.num_relations = len(self.relation2id)

        # 准备实体嵌入矩阵
        self.entity_emb_matrix = np.zeros((self.num_entities, embedding_dim))
        for entity, emb in entity_embeddings.items():
            idx = self.entity2id[entity]
            self.entity_emb_matrix[idx] = emb

        # 准备三元组数据
        self.triplets = []
        for h, r, t in triplets:
            if h in self.entity2id and t in self.entity2id and r in self.relation2id:
                h_idx = self.entity2id[h]
                t_idx = self.entity2id[t]
                r_idx = self.relation2id[r]
                self.triplets.append((h_idx, r_idx, t_idx))

        print(
            f"Loaded {len(self.triplets)} triplets for {self.num_entities} entities and {len(self.relation2id)} relations")

    def split_data(self, test_ratio=0.1):
        """划分训练集和测试集"""
        random.shuffle(self.triplets)
        split_idx = int(len(self.triplets) * (1 - test_ratio))
        train_data = self.triplets[:split_idx]
        test_data = self.triplets[split_idx:]
        return train_data, test_data

    def generate_negative_samples(self, triplets, num_negatives=1):
        """为每个正样本生成负样本"""
        negative_triplets = []
        for h, r, t in triplets:
            # 负样本：替换尾实体
            for _ in range(num_negatives):
                neg_t = random.randint(0, self.num_entities - 1)
                while neg_t == t:
                    neg_t = random.randint(0, self.num_entities - 1)
                negative_triplets.append((h, r, neg_t))
        return negative_triplets

    def train(self, epochs=100, batch_size=128, lr=0.001, lambda_cluster=0.1):
        """训练知识补全模型（添加聚类损失）"""
        # 划分数据集
        train_triplets, test_triplets = self.split_data(test_ratio=0.1)

        # 生成负样本
        neg_train = self.generate_negative_samples(train_triplets, num_negatives=1)
        neg_test = self.generate_negative_samples(test_triplets, num_negatives=1)

        # 合并正负样本
        all_train = train_triplets + neg_train
        all_test = test_triplets + neg_test

        # 创建标签：正样本为1，负样本为0
        train_labels = [1] * len(train_triplets) + [0] * len(neg_train)
        test_labels = [1] * len(test_triplets) + [0] * len(neg_test)

        # 转换为张量
        train_triplets_tensor = torch.LongTensor(all_train)
        train_labels_tensor = torch.FloatTensor(train_labels)
        test_triplets_tensor = torch.LongTensor(all_test)
        test_labels_tensor = torch.FloatTensor(test_labels)

        # 初始化模型 - 添加num_clusters参数
        model = SACN(
            num_entities=self.num_entities,
            num_relations=len(self.relation2id),
            embedding_dim=self.embedding_dim,
            init_embeddings=self.entity_emb_matrix,
            num_clusters=self.num_clusters  # 传递聚类数量
        ).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 训练循环
        best_score = 0
        for epoch in range(epochs):
            model.train()
            total_kc_loss = 0
            total_kl_loss = 0

            # 批量训练
            for i in range(0, len(train_triplets_tensor), batch_size):
                batch_triplets = train_triplets_tensor[i:i + batch_size].to(self.device)
                batch_labels = train_labels_tensor[i:i + batch_size].to(self.device)

                # 获取头实体和关系
                h = batch_triplets[:, 0]
                r = batch_triplets[:, 1]

                # 前向传播 - 获取知识补全预测和聚类分布
                kc_pred, cluster_q = model(h, r, return_cluster=True)

                # 获取尾实体的预测概率
                t = batch_triplets[:, 2]
                pred_t = kc_pred[torch.arange(len(t)), t]

                # 计算知识补全损失
                kc_loss = model.loss(pred_t, batch_labels)

                # 计算聚类损失
                kl_loss = model.kl_loss(cluster_q)

                # 总损失 = 知识补全损失 + λ * 聚类损失
                total_loss = kc_loss + lambda_cluster * kl_loss

                total_kc_loss += kc_loss.item()
                total_kl_loss += kl_loss.item()

                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            # 评估
            train_kc_loss = total_kc_loss / (len(train_triplets_tensor) / batch_size)
            train_kl_loss = total_kl_loss / (len(train_triplets_tensor) / batch_size)
            test_metrics = self.evaluate(model, test_triplets_tensor, test_labels_tensor, batch_size)

            print(f"Epoch {epoch + 1}/{epochs}: "
                  f"KC Loss: {train_kc_loss:.4f}, KL Loss: {train_kl_loss:.4f}, "
                  f"Test Acc: {test_metrics['accuracy']:.4f}, "
                  f"Test AUC: {test_metrics['auc']:.4f}, "
                  f"Test F1: {test_metrics['f1']:.4f}")

            # 保存最佳模型
            if test_metrics['f1'] > best_score:
                best_score = test_metrics['f1']
                torch.save(model.state_dict(), 'best_sacn_model.pth')

        print("Training completed. Best model saved as 'best_sacn_model.pth'")
        return model

    def evaluate(self, model, triplets_tensor, labels_tensor, batch_size=128):
        """评估模型性能"""
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for i in range(0, len(triplets_tensor), batch_size):
                batch_triplets = triplets_tensor[i:i + batch_size].to(self.device)
                batch_labels = labels_tensor[i:i + batch_size].to(self.device)

                h = batch_triplets[:, 0]
                r = batch_triplets[:, 1]
                t = batch_triplets[:, 2]

                pred = model(h, r)
                pred_t = pred[torch.arange(len(t)), t]

                all_preds.append(pred_t.cpu())
                all_labels.append(batch_labels.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # 计算评估指标
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

        # 将概率转换为二进制预测
        binary_preds = (all_preds > 0.5).astype(int)

        accuracy = accuracy_score(all_labels, binary_preds)
        auc = roc_auc_score(all_labels, all_preds)
        f1 = f1_score(all_labels, binary_preds)

        return {
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1
        }

    def predict_missing_links(self, model, head_entity, relation, top_k=10):
        """预测缺失的链接"""
        # 确保relation是有效的
        if isinstance(relation, str):
            if relation not in self.relation2id:
                raise ValueError(f"Unknown relation: {relation}")
            rel_idx = self.relation2id[relation]
        else:
            rel_idx = int(relation)
            if rel_idx >= self.num_relations:
                raise ValueError(f"Relation index {rel_idx} out of range")
        model.eval()

        # 获取实体索引
        head_idx = self.entity2id[head_entity]
        rel_idx = self.relation2id[relation]

        with torch.no_grad():
            # 转换为张量
            h_tensor = torch.LongTensor([head_idx]).to(self.device)
            r_tensor = torch.LongTensor([rel_idx]).to(self.device)

            # 预测所有尾实体的概率
            pred = model(h_tensor, r_tensor)
            pred = pred.squeeze().cpu().numpy()

            # 获取top-k预测
            top_indices = np.argsort(pred)[::-1][:top_k]

            # 转换为实体ID
            results = []
            for idx in top_indices:
                entity_id = self.id2entity[idx]
                score = pred[idx]
                results.append((entity_id, score))

        return results

    def cluster_entities(self, model, use_kmeans=True, n_clusters=None):
        """
        实体聚类方法
        Args:
            model: 训练好的SACN模型
            use_kmeans: 是否使用K-means聚类（False则使用模型自带的聚类层）
            n_clusters: 聚类数量（None则使用模型初始化的num_clusters）
        Returns:
            entity_clusters: 实体到聚类ID的映射字典
            cluster_centers: 聚类中心（仅当use_kmeans=True时返回）
        """
        model.eval()
        if n_clusters is None:
            n_clusters = model.num_clusters

        # 获取所有实体嵌入
        with torch.no_grad():
            all_entities = torch.arange(self.num_entities).to(self.device)
            entity_embeddings = model.emb_e(all_entities)

            if use_kmeans:
                # 使用K-means聚类
                from sklearn.cluster import KMeans
                embeddings_np = entity_embeddings.cpu().numpy()

                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_ids = kmeans.fit_predict(embeddings_np)
                cluster_centers = kmeans.cluster_centers_

                # 创建实体到聚类ID的映射
                entity_clusters = {
                    self.id2entity[i]: int(cluster_ids[i])
                    for i in range(self.num_entities)
                }

                return entity_clusters, cluster_centers
            else:
                # 使用模型自带的聚类层
                cluster_logits = model.cluster_fc(entity_embeddings)
                cluster_ids = torch.argmax(cluster_logits, dim=1)

                # 创建实体到聚类ID的映射
                entity_clusters = {
                    self.id2entity[i]: int(cluster_ids[i])
                    for i in range(self.num_entities)
                }

                return entity_clusters, None

    def add_similar_to_relations(self, model, output_file, cluster_threshold=0.8, intra_cluster_only=True):
        """
        根据聚类结果为实体添加similar_to关系

        Args:
            model: 训练好的SACN模型
            output_file: 输出文件路径
            cluster_threshold: 相似度阈值(0-1)
            intra_cluster_only: 是否只添加同一聚类内的相似关系
        """
        # 获取所有实体聚类结果
        entity_clusters, _ = self.cluster_entities(model, use_kmeans=True)

        # 获取所有实体嵌入
        with torch.no_grad():
            all_entities = torch.arange(self.num_entities).to(self.device)
            entity_embeddings = model.emb_e(all_entities).cpu().numpy()

        # 创建实体ID到索引的映射
        entity_id_to_idx = {self.id2entity[i]: i for i in range(self.num_entities)}

        # 计算实体间的余弦相似度
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(entity_embeddings)

        # 收集similar_to关系
        similar_relations = set()  # 使用集合避免重复

        # 遍历所有实体对
        for i in range(self.num_entities):
            entity_i = self.id2entity[i]
            cluster_i = entity_clusters.get(entity_i, -1)

            for j in range(i + 1, self.num_entities):
                entity_j = self.id2entity[j]
                cluster_j = entity_clusters.get(entity_j, -1)

                # 如果设置只考虑同一聚类内的关系，且实体不在同一聚类则跳过
                if intra_cluster_only and cluster_i != cluster_j:
                    continue

                # 获取相似度
                sim_score = similarity_matrix[i][j]

                # 如果相似度超过阈值，则添加关系
                if sim_score >= cluster_threshold:
                    # 确保关系是有序的，避免重复
                    if entity_i < entity_j:
                        similar_relations.add((entity_i, "similar_to", entity_j, sim_score))
                    else:
                        similar_relations.add((entity_j, "similar_to", entity_i, sim_score))

        # 将关系写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            for rel in similar_relations:
                f.write(f"{rel[0]}---{rel[1]}---{rel[2]}\n")

        print(f"Generated {len(similar_relations)} similar_to relations, saved to {output_file}")

    def is_patent(self, entity_id):
        # 规则1：检查是否在预加载的专利ID集合中
        # 规则2：检查是否符合专利ID格式（可选）
        return entity_id in self.patent_ids

    def get_kg_structure(self, entity_id, max_relations=50, depth=1):
        """
        获取多层知识图谱结构数据
        Args:
            entity_id: 实体ID或专利ID
            max_relations: 每层最多返回的关系数量
            depth: 查询深度（默认为1，只返回直接关联）
        Returns:
            dict: 包含节点和边的知识图谱结构数据
        """
        # 初始化数据结构
        kg_data = {"nodes": [], "links": []}
        visited_entities = set()  # 记录已访问的实体，避免循环
        
        def _dfs(current_entity, current_depth):
            if current_depth > depth or current_entity in visited_entities:
                return
            
            visited_entities.add(current_entity)
            
            # 获取当前实体的标准化ID和索引
            if self.is_patent(current_entity):
                entity_idx = self.entity2id[current_entity]
                display_id = f"CN{entity_idx + 100000}"
                entity_type = "patent"
            else:
                entity_idx = self.entity2id[current_entity]
                display_id = current_entity
                entity_type = "entity"
            
            # 添加当前节点
            kg_data["nodes"].append({
                "id": display_id,
                "type": entity_type,
                "label": current_entity,
                "depth": current_depth  # 记录层级（可选）
            })
            
            # 获取直接关联的三元组（限制数量）
            related_triplets = [
                (h, r, t) for h, r, t in self.triplets 
                if h == entity_idx or t == entity_idx
            ][:max_relations]
            print("related_triplets",related_triplets)
            # 处理关联关系
            for h, r, t in related_triplets:
                if h == entity_idx:  # 当前实体是头实体
                    other_entity = self.id2entity[t]
                    link = {
                        "source": display_id,
                        "target": other_entity,
                        "relation": self.id2relation[r],
                        "depth": current_depth
                    }
                    kg_data["links"].append(link)
                    _dfs(other_entity, current_depth + 1)  # 递归下一层
                    
                elif t == entity_idx:  # 当前实体是尾实体
                    other_entity = self.id2entity[h]
                    link = {
                        "source": other_entity,
                        "target": display_id,
                        "relation": self.id2relation[r],
                        "depth": current_depth
                    }
                    kg_data["links"].append(link)
                    _dfs(other_entity, current_depth + 1)  # 递归下一层
        
        entity_type = "patent" if self.is_patent(entity_id) else "entity"
        if entity_type == "patent":
            entity_id = self.id2entity[int(entity_id[2:])-100000]

        if entity_id not in self.entity2id:
            return {"error": "Entity not found"}
            
        # 开始递归遍历
        _dfs(entity_id, 0)
        
        # 去重（虽然visited_entities已避免循环，但可能有多条路径到达同一节点）
        kg_data["nodes"] = list({node["id"]: node for node in kg_data["nodes"]}.values())
        kg_data["links"] = list({(link["source"], link["target"], link["relation"]): link for link in kg_data["links"]}.values())
        
        return kg_data
    
    def predict_relation(self, model, head_entity, tail_entity, top_k=5):
        """
        预测头实体和尾实体之间可能存在的关系
        Args:
            model: 训练好的模型
            head_entity: 头实体ID
            tail_entity: 尾实体ID
            top_k: 返回前k个最可能的关系
        Returns:
            list: 可能的关系列表，每个元素为(关系, 概率)
        """
        # 获取实体索引
        head_idx = self.entity2id[head_entity]
        tail_idx = self.entity2id[tail_entity]
        
        model.eval()
        results = []
        
        # 遍历所有关系
        for rel in self.relation2id.keys():
            rel_idx = self.relation2id[rel]
            
            with torch.no_grad():
                h_tensor = torch.LongTensor([head_idx]).to(self.device)
                r_tensor = torch.LongTensor([rel_idx]).to(self.device)
                
                # 预测所有尾实体的概率
                pred = model(h_tensor, r_tensor)  # (1, num_entities)
                pred_t = pred[0, tail_idx].item()  # 目标尾实体的概率
                
                results.append((rel, pred_t))
        
        # 按概率排序并返回top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def predict_path(self, model, head_entity, tail_entity, max_depth=3, beam_size=5, threshold=0.5):
        """
        预测从头实体到尾实体的路径（使用beam search）
        Args:
            model: 训练好的模型
            head_entity: 头实体ID
            tail_entity: 尾实体ID
            max_depth: 最大路径深度（跳数）
            beam_size: beam search的宽度
            threshold: 路径概率阈值
        Returns:
            list: 路径列表，每个元素为(path, path_probability)
        """
        head_idx = self.entity2id[head_entity]
        tail_idx = self.entity2id[tail_entity]
        
        # 初始化beam: (current_entity, path, path_prob)
        # path格式: [head_entity, relation1, entity1, relation2, entity2, ...]
        beam = [(head_idx, [head_entity], 1.0)]
        completed_paths = []
        
        for depth in range(max_depth):
            new_beam = []
            
            for current_idx, path, path_prob in beam:
                # 获取当前实体的所有可能关系
                for rel in self.relation2id.keys():
                    rel_idx = self.relation2id[rel]
                    
                    with torch.no_grad():
                        h_tensor = torch.LongTensor([current_idx]).to(self.device)
                        r_tensor = torch.LongTensor([rel_idx]).to(self.device)
                        
                        # 预测所有尾实体的概率
                        pred = model(h_tensor, r_tensor).squeeze().cpu().numpy()
                        
                        # 获取概率最高的实体
                        top_indices = np.argsort(pred)[::-1][:beam_size]
                        
                        for idx in top_indices:
                            prob = pred[idx]
                            new_prob = path_prob * prob
                            
                            # 跳过概率过低的路径
                            if new_prob < threshold:
                                continue
                            
                            new_entity = self.id2entity[idx]
                            new_path = path + [rel, new_entity]
                            
                            # 如果到达目标实体
                            if idx == tail_idx:
                                completed_paths.append((new_path, new_prob))
                            else:
                                new_beam.append((idx, new_path, new_prob))
            
            # 如果没有新的候选路径，提前终止
            if not new_beam:
                break
                
            # 按路径概率排序并保留top beam_size
            new_beam.sort(key=lambda x: x[2], reverse=True)
            beam = new_beam[:beam_size]
        
        # 按路径概率排序并返回
        completed_paths.sort(key=lambda x: x[1], reverse=True)
        return completed_paths

    def evaluate_clustering(self, entity_clusters, true_labels=None):
        """
        评估聚类效果 (ACC, NMI, ARI)
        Args:
            entity_clusters: 实体到聚类ID的映射字典
            true_labels: 实体真实标签字典 (entity->label)
        Returns:
            metrics: 包含ACC, NMI, ARI的字典
        """
        if true_labels is None:
            print("Warning: True labels not provided. Only clustering assignments returned.")
            return {}

        # 准备标签数组
        pred_labels = []
        true_label_list = []

        for entity, cluster_id in entity_clusters.items():
            if entity in true_labels:
                pred_labels.append(cluster_id)
                true_label_list.append(true_labels[entity])

        pred_labels = np.array(pred_labels)
        true_label_list = np.array(true_label_list)

        # 计算NMI和ARI
        nmi = normalized_mutual_info_score(true_label_list, pred_labels)
        ari = adjusted_rand_score(true_label_list, pred_labels)

        # 计算ACC (需要最优匹配)
        contingency = contingency_matrix(true_label_list, pred_labels)
        cost_matrix = contingency.max() - contingency
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        acc = contingency[row_ind, col_ind].sum() / np.sum(contingency)

        return {
            'ACC': acc,
            'NMI': nmi,
            'ARI': ari
        }

    def visualize_clusters(self, model, entity_clusters, cluster_centers=None,
                           sample_size=500, random_state=42, method='pca',
                           save_path=None, dpi=300):
        """
        可视化聚类结果并保存图片
        Args:
            model: 训练好的SACN模型
            entity_clusters: 实体聚类结果
            cluster_centers: 聚类中心（可选）
            sample_size: 可视化采样的实体数量
            random_state: 随机种子
            method: 降维方法 ('pca' 或 'tsne')
            save_path: 图片保存路径（None则不保存）
            dpi: 图片分辨率
        Returns:
            fig: matplotlib的Figure对象
        """
        try:
            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE
            from sklearn.decomposition import PCA
        except ImportError:
            print("Visualization requires matplotlib and scikit-learn")
            return None

        # 获取实体嵌入和聚类标签
        all_embeddings = []
        all_labels = []
        entity_ids = list(entity_clusters.keys())

        # 随机采样（如果实体数量太多）
        if len(entity_ids) > sample_size:
            import random
            random.seed(random_state)
            entity_ids = random.sample(entity_ids, sample_size)

        with torch.no_grad():
            for entity in entity_ids:
                idx = self.entity2id[entity]
                embedding = model.emb_e(torch.LongTensor([idx]).to(self.device))
                all_embeddings.append(embedding.cpu().numpy())
                all_labels.append(entity_clusters[entity])

        embeddings_np = np.concatenate(all_embeddings, axis=0)
        labels_np = np.array(all_labels)

        # 降维处理
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=random_state)
            embeddings_2d = reducer.fit_transform(embeddings_np)

            if cluster_centers is not None:
                combined = np.vstack([embeddings_np, cluster_centers])
                combined_2d = reducer.fit_transform(combined)
                centers_2d = combined_2d[-len(cluster_centers):]
                embeddings_2d = combined_2d[:-len(cluster_centers)]
        else:
            reducer = PCA(n_components=2, random_state=random_state)
            embeddings_2d = reducer.fit_transform(embeddings_np)

            if cluster_centers is not None:
                centers_2d = reducer.transform(cluster_centers)

        # 创建图形
        fig = plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=labels_np, cmap='tab20', alpha=0.6
        )

        # 标记聚类中心
        if cluster_centers is not None:
            plt.scatter(
                centers_2d[:, 0], centers_2d[:, 1],
                c='red', marker='x', s=100, linewidths=2,
                label='Cluster Centers'
            )
            plt.legend()

        plt.title(f'Entity Clustering Visualization ({method.upper()})')
        plt.colorbar(scatter, label='Cluster ID')

        # 保存图片
        if save_path is not None:
            try:
                plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
                print(f"Cluster visualization saved to {save_path}")
            except Exception as e:
                print(f"Could not save visualization: {str(e)}")

        plt.show()
        return fig


def load_true_labels(file_path):
    """加载实体真实标签"""
    true_labels = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                entity = parts[0]
                label = int(parts[1])
                true_labels[entity] = label
    return true_labels


def load_refined_embeddings(file_path):
    """加载强化学习后的实体嵌入"""
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                patent_id = parts[0]
                embedding = np.array([float(x) for x in parts[1].split()], dtype=np.float32)
                embeddings[patent_id] = embedding
    return embeddings


def load_triplets_from_files(files):
    """从文件中加载三元组数据"""
    triplets = []
    for file_name in files:
        # 检查文件是否存在
        if not os.path.exists(file_name):
            print(f"Warning: File {file_name} not found. Skipping.")
            continue

        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('---')
                    if len(parts) != 3:
                        continue
                    patent = parts[0].strip()
                    relation = parts[1].strip()
                    value = parts[2].strip()

                    # 处理单个值或多个值的情况
                    if value.startswith('[') and value.endswith(']'):
                        try:
                            values = eval(value)
                            for v in values:
                                clean_value = str(v).strip("'\"")
                                triplets.append((patent, relation, clean_value))
                        except:
                            continue
                    else:
                        clean_value = value.strip("'\"")
                        triplets.append((patent, relation, clean_value))
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
    return triplets


def extract_patent_titles(document_text):
    # 存储唯一专利标题的集合
    unique_titles = set()

    # 按行分割文档
    lines = document_text.strip().split('\n')

    for line in lines:
        # 检查是否包含标题分隔符
        if "---titleKey---" in line:
            # 分割出专利标题部分
            title_part = line.split("---titleKey---")[0].strip()
            # 添加到集合（自动去重）
            unique_titles.add(title_part)

    # 将唯一标题写入文件
    with open("patent_titles.txt", "w", encoding="utf-8") as f:
        for title in sorted(unique_titles):  # 按字母排序
            f.write(title + "\n")


from sklearn.cluster import KMeans


def generate_cluster_labels(embeddings_file, output_file, n_clusters=10):
    """基于实体嵌入生成聚类标签"""
    embeddings = {}

    # 加载嵌入
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                patent_id = parts[0]
                embedding = np.array([float(x) for x in parts[1].split()])
                embeddings[patent_id] = embedding

    # 准备数据
    patent_ids = list(embeddings.keys())
    embedding_matrix = np.array([embeddings[pid] for pid in patent_ids])

    # 使用KMeans聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embedding_matrix)

    # 保存标签文件
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for pid, label in zip(patent_ids, cluster_labels):
            f_out.write(f"{pid}\t{label}\n")

    print(f"Generated cluster-based labels for {len(patent_ids)} patents")


def main():
    # 1. 加载强化学习后的实体嵌入
    refined_embeddings = load_refined_embeddings('refined_patent_embeddings.txt')
    print(f"Loaded {len(refined_embeddings)} refined embeddings")

    # 2. 加载三元组数据
    files = [
        'new_title_triplets.txt'
    ]
    triplets = load_triplets_from_files(files)
    print(f"Loaded {len(triplets)} triplets")

    # 检查是否有足够的数据
    if len(triplets) == 0:
        print("Error: No triplets loaded. Exiting.")
        return

    # 3. 初始化知识补全模块
    num_clusters = 10  # 设置聚类数量
    # 初始化知识补全模块时使用正确的relation数量
    kc = KnowledgeCompletion(
        entity_embeddings=refined_embeddings,
        triplets=triplets,
        num_relations=5,  # 包含similar_to关系
        embedding_dim=len(next(iter(refined_embeddings.values()))),
        num_clusters=num_clusters
    )

    # 4. 训练知识补全模型
    model = kc.train(epochs=100, batch_size=256, lr=0.001, lambda_cluster=0.1)  # lr=0.01

    with open("new_title_triplets.txt", "r", encoding="utf-8") as file:
        document = file.read()
    extract_patent_titles(document)

    # 生成真实标签（选择一种方法）
    generate_cluster_labels('refined_patent_embeddings.txt', 'true_labels.txt', n_clusters=5)
    # 加载真实标签
    true_labels = {}
    with open('true_labels.txt', 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                entity = parts[0]
                label = int(parts[1])
                true_labels[entity] = label

    # 5. 执行实体聚类
    print("\nPerforming entity clustering...")
    entity_clusters, cluster_centers = kc.cluster_entities(model, use_kmeans=True)

    # 评估聚类效果
    if true_labels:
        metrics = kc.evaluate_clustering(entity_clusters, true_labels)
        print("\nClustering Evaluation Metrics:")
        print(f"ACC: {metrics['ACC']:.4f}")
        print(f"NMI: {metrics['NMI']:.4f}")
        print(f"ARI: {metrics['ARI']:.4f}")

    # 打印部分聚类结果
    print("\nSample clustering results:")
    sample_entities = random.sample(list(entity_clusters.keys()), min(10, len(entity_clusters)))
    for entity in sample_entities:
        print(f"Entity: {entity} -> Cluster {entity_clusters[entity]}")

    # 可视化聚类结果
    try:
        kc.visualize_clusters(
            model,
            entity_clusters,
            cluster_centers,
            method='tsne',
            save_path='cluster_visualization_tsne.png',
            dpi=300
        )
    except Exception as e:
        print(f"Could not visualize clusters: {str(e)}")

    # 保存聚类结果
    with open('entity_clusters.txt', 'w') as f:
        for entity, cluster_id in entity_clusters.items():
            f.write(f"{entity}\t{cluster_id}\n")
    print("\nClustering results saved to entity_clusters.txt")

    # 6. 生成similar_to关系
    print("\nGenerating similar_to relations based on clustering...")
    kc.add_similar_to_relations(
        model,
        output_file='similar_to_relations.txt',
        cluster_threshold=0.6,  # 可以根据需要调整
        intra_cluster_only=True  # 只添加同一聚类内的相似关系
    )

    # 7. 重新加载包含新关系的知识图谱
    print("\nReloading knowledge graph with similar_to relations...")
    files = [
        'new_title_triplets.txt',
        'similar_to_relations.txt'
    ]
    triplets_with_similar = load_triplets_from_files(files)

    # 使用增强后的三元组重新初始化
    enhanced_kc = KnowledgeCompletion(
        entity_embeddings=refined_embeddings,
        triplets=triplets_with_similar,
        num_relations=5,  # 增加了similar_to关系
        embedding_dim=len(next(iter(refined_embeddings.values()))),
        num_clusters=num_clusters
    )

    print("Knowledge graph enhancement completed.")

    # 8. 训练知识补全模型（使用新的关系数量）
    model = enhanced_kc.train(epochs=100, batch_size=256, lr=0.001, lambda_cluster=0.1)

    # 保存新模型
    torch.save(model.state_dict(), 'best_sacn_model_v2.pth')


if __name__ == '__main__':
    main()
