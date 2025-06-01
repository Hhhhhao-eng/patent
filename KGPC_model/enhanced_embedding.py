import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import random
from torch.distributions import Categorical


class PatentEnvironment:
    """GPU优化的专利表示学习环境"""

    def __init__(self, patent_embeddings, patent_data, device, k=5):
        """
        Args:
            patent_embeddings: 字典 {专利ID: GPU张量}
            patent_data: 原始专利数据
            device: torch设备对象
            k: 每个正样本的负样本数
        """
        self.device = device

        # 确保嵌入为float32类型
        self.embeddings = {k: v.to(device).to(torch.float32) for k, v in patent_embeddings.items()}
        self.patent_data = patent_data
        self.k = k
        self.patent_ids = list(patent_embeddings.keys())

        # 预计算有效负样本候选池
        self.valid_patents_set = set(self.patent_ids)
        self.negative_candidates = {}
        for patent_id in self.patent_ids:
            if patent_id in patent_data:
                # 获取当前专利的所有关系值
                related_values = set()
                for rel_type in ['titleKey', 'bgKey', 'clKey', 'patentee']:
                    if rel_type in patent_data[patent_id]:
                        related_values.update(patent_data[patent_id][rel_type])
                # 创建负样本候选池
                candidate_pool = self.valid_patents_set - related_values
                # 只保留存在的嵌入
                self.negative_candidates[patent_id] = [v for v in candidate_pool if v in self.embeddings]

        # 关系类型映射
        self.relation_types = ['titleKey', 'bgKey', 'clKey', 'patentee']
        self.relation_to_idx = {rel: idx + 1 for idx, rel in enumerate(self.relation_types)}

        # 预构建样本
        self.positive_samples = self._build_relation_samples()
        self.current_sample_idx = 0

        print(f"Environment initialized with {len(self.positive_samples)} positive samples")

    def _build_relation_samples(self):
        """构建GPU友好的样本列表"""
        samples = []
        for patent, relations in self.patent_data.items():
            if patent not in self.embeddings:
                continue
            for rel_type, values in relations.items():
                for value in values:
                    if value in self.embeddings:
                        samples.append((patent, value, rel_type))
        return samples

    def reset(self):
        """重置环境"""
        self.current_sample_idx = 0
        random.shuffle(self.positive_samples)
        return self._get_current_state()

    def _get_current_state(self):
        """获取当前状态(已放在GPU上)"""
        if self.current_sample_idx >= len(self.positive_samples):
            return None

        patent, value, rel_type = self.positive_samples[self.current_sample_idx]
        return {
            'patent_emb': self.embeddings[patent],
            'value_emb': self.embeddings[value],
            'rel_type': self.relation_to_idx[rel_type]
        }

    def step(self, action):
        """执行动作并返回GPU张量"""
        if self.current_sample_idx >= len(self.positive_samples):
            return None, torch.tensor(0.0, device=self.device, dtype=torch.float32), True

        patent, value, rel_type = self.positive_samples[self.current_sample_idx]

        if action == 0:  # 正样本
            reward = F.cosine_similarity(
                self.embeddings[patent].unsqueeze(0),
                self.embeddings[value].unsqueeze(0),
                dim=1
            ).squeeze()
        else:  # 负样本
            # 从预计算的负样本池中随机选择
            if patent in self.negative_candidates and self.negative_candidates[patent]:
                neg_value = random.choice(self.negative_candidates[patent])
            else:
                # 回退机制：随机选择任何专利
                neg_value = random.choice(self.patent_ids)
                while neg_value == patent or neg_value not in self.embeddings:
                    neg_value = random.choice(self.patent_ids)

            reward = -F.cosine_similarity(
                self.embeddings[patent].unsqueeze(0),
                self.embeddings[neg_value].unsqueeze(0),
                dim=1
            ).squeeze()

        self.current_sample_idx += 1
        next_state = self._get_current_state()
        done = next_state is None

        return next_state, reward, done


class RepresentationPolicy(nn.Module):
    """GPU优化的策略网络"""

    def __init__(self, embedding_dim, num_relations, device):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim

        # 关系嵌入层
        self.relation_emb = nn.Embedding(num_relations + 1, embedding_dim, padding_idx=0).to(device)

        # 策略网络
        self.policy_net = nn.Sequential(
            nn.Linear(embedding_dim * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        ).to(device)

        # 价值网络
        self.value_net = nn.Sequential(
            nn.Linear(embedding_dim * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device)

    def forward(self, patent_emb, value_emb, rel_type):
        rel_emb = self.relation_emb(rel_type)
        x = torch.cat([patent_emb, value_emb, rel_emb], dim=-1)
        policy_logits = self.policy_net(x)
        value = self.value_net(x)
        return policy_logits, value


class ReinforcedRepresentationLearner:
    """GPU优化的强化学习器"""

    def __init__(self, patent_embeddings, patent_data):
        # 自动检测GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 初始化环境
        self.env = PatentEnvironment(patent_embeddings, patent_data, self.device)

        # 获取嵌入维度
        sample_emb = next(iter(patent_embeddings.values()))
        embedding_dim = sample_emb.shape[0] if isinstance(sample_emb, torch.Tensor) else len(sample_emb)

        print(f"Embedding dimension: {embedding_dim}")

        # 初始化策略网络
        self.policy = RepresentationPolicy(embedding_dim, 4, self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
        self.gamma = 0.99

        # 训练统计
        self.loss_history = []
        self.reward_history = []

    def train(self, episodes=1000, batch_size=32):
        """GPU加速的训练循环"""
        self.policy.train()

        for episode in range(episodes):
            batch = {
                'rewards': [],
                'log_probs': [],
                'values': [],
                'entropies': []
            }

            state = self.env.reset()

            while state is not None:
                # 前向传播
                action_probs, value = self.policy(
                    state['patent_emb'].unsqueeze(0),
                    state['value_emb'].unsqueeze(0),
                    torch.tensor([state['rel_type']], device=self.device)
                )

                # 采样动作
                dist = Categorical(F.softmax(action_probs, dim=-1))
                action = dist.sample()

                # 环境交互
                next_state, reward, done = self.env.step(action.item())

                # 存储结果
                batch['rewards'].append(reward)
                batch['log_probs'].append(dist.log_prob(action))
                batch['values'].append(value.squeeze())
                batch['entropies'].append(dist.entropy())

                state = next_state

            # 检查是否有数据
            if not batch['rewards']:
                print(f"No data collected in episode {episode}")
                continue

            # 计算折扣回报
            rewards = batch['rewards']
            returns = []
            R = torch.tensor(0.0, device=self.device, dtype=torch.float32)

            for r in reversed(rewards):
                R = r + self.gamma * R
                returns.insert(0, R)

            try:
                returns = torch.stack(returns)
                if returns.std() == 0:
                    returns = (returns - returns.mean())
                else:
                    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

                # 转换为张量
                rewards = torch.stack(batch['rewards'])
                log_probs = torch.stack(batch['log_probs'])
                values = torch.stack(batch['values'])
                entropies = torch.stack(batch['entropies'])

                # 计算优势
                advantages = returns - values.detach()

                # 计算损失
                policy_loss = -(log_probs * advantages.detach()).mean()
                value_loss = F.mse_loss(values, returns)
                entropy_bonus = entropies.mean()  # 正熵，增加探索

                total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus

                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                # 记录统计
                self.loss_history.append(total_loss.item())
                self.reward_history.append(returns.mean().item())

                if episode % 10 == 0:
                    print(f"Episode {episode}: Loss={total_loss.item():.4f}, "
                          f"Avg Reward={returns.mean().item():.4f}, "
                          f"Policy Loss={policy_loss.item():.4f}, "
                          f"Value Loss={value_loss.item():.4f}")

            except Exception as e:
                print(f"Error in episode {episode}: {str(e)}")
                continue

    def update_embeddings(self):
        """GPU加速的嵌入更新"""
        updated_embeddings = {}
        self.policy.eval()

        with torch.no_grad():
            for patent_id, emb in self.env.embeddings.items():
                relations = self.env.patent_data.get(patent_id, {})
                update = torch.zeros_like(emb)

                for rel_type, values in relations.items():
                    if rel_type not in self.env.relation_types:
                        continue

                    rel_idx = self.env.relation_to_idx[rel_type]
                    rel_tensor = torch.tensor([rel_idx], device=self.device)

                    for value in values:
                        if value not in self.env.embeddings:
                            continue

                        # GPU上的前向计算
                        action_logits, _ = self.policy(
                            emb.unsqueeze(0),
                            self.env.embeddings[value].unsqueeze(0),
                            rel_tensor
                        )

                        # 获取正样本动作概率
                        action_probs = F.softmax(action_logits, dim=-1)
                        positive_prob = action_probs[0, 0].item()

                        # 更新嵌入
                        update += positive_prob * self.env.embeddings[value]

                updated_emb = emb + 0.1 * update
                updated_embeddings[patent_id] = updated_emb.cpu().numpy()  # 转为numpy数组保存

        return updated_embeddings


def load_patent_embeddings(file_path, device):
    """加载嵌入并自动放到指定设备"""
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                patent_id = parts[0]
                embedding = np.array([float(x) for x in parts[1].split()], dtype=np.float32)
                embeddings[patent_id] = torch.tensor(embedding, device=device, dtype=torch.float32)
    return embeddings


def main():
    # 加载专利数据
    patent_data = defaultdict(lambda: {
        'titleKey': set(),
        'bgKey': set(),
        'clKey': set(),
        'patentee': set()
    })

    files = [
        'new_title_triplets.txt',
        'new_bg_triplets.txt',
        'new_cl_triplets.txt',
        'new_patentee_triplets.txt'
    ]

    for file_name in files:
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
                                patent_data[patent][relation].add(clean_value)
                        except:
                            continue
                    else:
                        clean_value = value.strip("'\"")
                        patent_data[patent][relation].add(clean_value)
        except FileNotFoundError:
            print(f"Warning: File {file_name} not found. Skipping.")
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")

    print(f"Loaded patent data: {len(patent_data)} patents")

    # 加载嵌入并自动选择GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patent_embeddings = {}
    try:
        patent_embeddings = load_patent_embeddings('patent_embeddings.txt', device)
    except FileNotFoundError:
        print("Error: patent_embeddings.txt not found.")
        return

    print(f"Loaded embeddings: {len(patent_embeddings)} embeddings")

    # 过滤无效数据
    valid_patents = set(patent_embeddings.keys())
    filtered_data = {
        p: {r: [v for v in vals if v in valid_patents]
            for r, vals in rels.items()}
        for p, rels in patent_data.items()
        if p in valid_patents and any(len(v) > 0 for v in rels.values())
    }

    print(f"有效专利数: {len(filtered_data)}")

    # 训练
    learner = ReinforcedRepresentationLearner(patent_embeddings, filtered_data)
    learner.train(episodes=500)

    # 保存结果
    refined_embeddings = learner.update_embeddings()
    with open('refined_patent_embeddings.txt', 'w', encoding='utf-8') as f:
        for patent, emb in refined_embeddings.items():
            emb_str = ' '.join([f"{x:.6f}" for x in emb])
            f.write(f"{patent}\t{emb_str}\n")


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