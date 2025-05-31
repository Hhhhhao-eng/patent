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


class SACN(nn.Module):
    """基于强化实体表征的知识图谱补全模型"""

    def __init__(self, num_entities, num_relations, embedding_dim, init_embeddings,
                 input_dropout=0.1, dropout_rate=0.2, channels=32, kernel_size=3):
        super(SACN, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations

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

        # 卷积层 - 修复维度问题
        self.conv1 = nn.Conv1d(2, channels, kernel_size, stride=1, padding=int(math.floor(kernel_size / 2)))

        # 批归一化层 - 修复维度问题
        self.bn0 = nn.BatchNorm1d(2)  # 输入通道数=2
        self.bn1 = nn.BatchNorm1d(channels)  # 卷积输出通道数
        self.bn2 = nn.BatchNorm1d(embedding_dim)  # 全连接层输出维度

        # 偏置项
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))

        # 全连接层
        self.fc = nn.Linear(embedding_dim * channels, embedding_dim)

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        """初始化权重"""
        nn.init.xavier_normal_(self.emb_rel.weight.data)
        nn.init.xavier_normal_(self.conv1.weight.data)
        nn.init.xavier_normal_(self.fc.weight.data)

    def forward(self, e1, rel):
        """
        前向传播
        Args:
            e1: 头实体索引 (batch_size,)
            rel: 关系索引 (batch_size,)
        Returns:
            pred: 预测的概率分布 (batch_size, num_entities)
        """
        batch_size = e1.shape[0]

        # 获取实体和关系嵌入
        e1_embedded = self.emb_e(e1)  # (batch_size, embedding_dim)
        rel_embedded = self.emb_rel(rel)  # (batch_size, embedding_dim)

        # 拼接实体和关系嵌入 - 修复维度问题
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

        # 计算所有实体的分数
        all_entities = self.emb_e.weight  # (num_entities, embedding_dim)
        x = torch.mm(x, all_entities.transpose(1, 0))  # (batch_size, num_entities)
        x += self.b.expand_as(x)  # 添加偏置

        # Sigmoid激活得到概率
        pred = torch.sigmoid(x)

        return pred


class KnowledgeCompletion:
    """知识图谱知识补全模块"""

    def __init__(self, entity_embeddings, triplets, num_relations, embedding_dim=32,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        # 创建实体到索引的映射
        self.entity2id = {entity: idx for idx, entity in enumerate(entity_embeddings.keys())}
        self.id2entity = {idx: entity for entity, idx in self.entity2id.items()}
        self.num_entities = len(self.entity2id)

        # 创建关系映射
        self.relation2id = {}
        self.id2relation = {}
        for _, rel, _ in triplets:
            if rel not in self.relation2id:
                rel_id = len(self.relation2id)
                self.relation2id[rel] = rel_id
                self.id2relation[rel_id] = rel

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

    def train(self, epochs=100, batch_size=128, lr=0.001):
        """训练知识补全模型"""
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

        # 初始化模型 - 修复维度问题
        model = SACN(
            num_entities=self.num_entities,
            num_relations=len(self.relation2id),
            embedding_dim=self.embedding_dim,
            init_embeddings=self.entity_emb_matrix
        ).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 训练循环
        best_score = 0
        for epoch in range(epochs):
            model.train()
            total_loss = 0

            # 批量训练
            for i in range(0, len(train_triplets_tensor), batch_size):
                batch_triplets = train_triplets_tensor[i:i + batch_size].to(self.device)
                batch_labels = train_labels_tensor[i:i + batch_size].to(self.device)

                # 获取头实体和关系
                h = batch_triplets[:, 0]
                r = batch_triplets[:, 1]

                # 前向传播
                pred = model(h, r)

                # 获取尾实体的预测概率
                t = batch_triplets[:, 2]
                pred_t = pred[torch.arange(len(t)), t]

                # 计算损失
                loss = model.loss(pred_t, batch_labels)
                total_loss += loss.item()

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 评估
            train_loss = total_loss / (len(train_triplets_tensor) / batch_size)
            test_metrics = self.evaluate(model, test_triplets_tensor, test_labels_tensor, batch_size)

            print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.4f}, "
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


def main():
    # 1. 加载强化学习后的实体嵌入
    refined_embeddings = load_refined_embeddings('refined_patent_embeddings.txt')
    print(f"Loaded {len(refined_embeddings)} refined embeddings")

    # 2. 加载三元组数据
    files = [
        'new_title_triplets.txt',
        'new_bg_triplets.txt',
        'new_cl_triplets.txt',
        'new_patentee_triplets.txt'
    ]
    triplets = load_triplets_from_files(files)
    print(f"Loaded {len(triplets)} triplets")

    # 检查是否有足够的数据
    if len(triplets) == 0:
        print("Error: No triplets loaded. Exiting.")
        return

    # 3. 初始化知识补全模块
    embedding_dim = len(next(iter(refined_embeddings.values())))
    kc = KnowledgeCompletion(
        entity_embeddings=refined_embeddings,
        triplets=triplets,
        num_relations=4,  # 根据实际关系类型设置
        embedding_dim=embedding_dim  # 添加嵌入维度参数
    )

    # 4. 训练知识补全模型
    model = kc.train(epochs=50, batch_size=256, lr=0.001)

    # 5. 示例：预测缺失链接
    if kc.entity2id and kc.relation2id:
        head_entity = next(iter(kc.entity2id.keys()))  # 使用第一个实体作为示例
        relation = next(iter(kc.relation2id.keys()))  # 使用第一个关系作为示例

        print(f"\nPredicting missing links for: {head_entity} - {relation}")
        predictions = kc.predict_missing_links(model, head_entity, relation, top_k=5)

        print(f"\nTop predictions for ({head_entity}, {relation}, ?):")
        for entity, score in predictions:
            print(f"  - {entity}: {score:.4f}")


if __name__ == '__main__':
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 运行主程序
    main()