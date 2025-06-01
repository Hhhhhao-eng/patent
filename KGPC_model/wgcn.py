import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
from collections import defaultdict


def load_patent_data(file_path):
    """从文件加载专利数据并分开存储"""
    patent_data = defaultdict(lambda: {
        'titleKey': set(),
        'bgKey': set(),
        'clKey': set(),
        'patentee': set()
    })

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 解析每行数据
            parts = line.split('---')
            if len(parts) != 3:
                continue

            patent = parts[0].strip()
            relation = parts[1].strip()
            value = parts[2].strip()

            # 处理值可能是列表的情况
            if value.startswith('[') and value.endswith(']'):
                try:
                    values = eval(value)
                    for v in values:
                        patent_data[patent][relation].add(str(v).strip("'"))
                except:
                    continue
            else:
                patent_data[patent][relation].add(value.strip("'"))

    return patent_data


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, num_relations, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.num_relations = num_relations
        self.alpha = nn.Embedding(num_relations + 1, 1, padding_idx=0)

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        alp = self.alpha(adj[1]).t()[0]
        A = torch.sparse_coo_tensor(adj[0], alp, torch.Size([adj[2], adj[2]]))
        A = A + A.transpose(0, 1)  # 使矩阵对称

        support = torch.mm(input, self.weight)
        output = torch.sparse.mm(A, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output


class WGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, num_relations, dropout=0.5):
        super(WGCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid, num_relations)
        self.gc2 = GraphConvolution(nhid, nclass, num_relations)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x  # 不进行softmax，直接返回嵌入


def build_graph(patent_data):
    """构建整个专利知识图谱"""
    # 收集所有节点
    patents = list(patent_data.keys())
    keywords = set()
    patentees = set()

    for patent, relations in patent_data.items():
        keywords.update(relations['titleKey'])
        keywords.update(relations['bgKey'])
        keywords.update(relations['clKey'])
        patentees.update(relations['patentee'])

    keywords = list(keywords)
    patentees = list(patentees)
    nodes = patents + keywords + patentees
    node2idx = {node: i for i, node in enumerate(nodes)}

    # 构建边
    edges = []
    edge_types = []
    relation2type = {'titleKey': 1, 'bgKey': 2, 'clKey': 3, 'patentee': 4}

    for patent, relations in patent_data.items():
        for rel_type, values in relations.items():
            for value in values:
                src = node2idx[patent]
                dst = node2idx[value]
                edges.append([src, dst])
                edge_types.append(relation2type[rel_type])

    # 转换为张量
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    num_nodes = len(nodes)

    # 初始化特征 (可以使用预训练的词向量)
    nfeat = 128
    x = torch.randn(num_nodes, nfeat)

    return (edge_index, edge_type, num_nodes), x, node2idx, patents


def train_wgcn(adj, features, num_relations):
    """训练WGCN模型"""
    nhid = 64
    nclass = 32
    dropout = 0.3
    epochs = 200

    model = WGCN(features.shape[1], nhid, nclass, num_relations, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        embeddings = model(features, adj)

        # 使用重建损失作为示例
        loss = F.mse_loss(embeddings, torch.randn_like(embeddings))
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item():.4f}')

    return model


def save_patent_embeddings(patent_list, patent_embeddings, output_file):
    """保存专利嵌入到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for patent, embedding in zip(patent_list, patent_embeddings):
            embedding_str = ' '.join([str(x.item()) for x in embedding])
            f.write(f"{patent}\t{embedding_str}\n")


def main():
    # 加载数据
    file_path = 'new_title_triplets.txt'
    patent_data = load_patent_data(file_path)
    print(patent_data[1])
    print(f"总共加载了 {len(patent_data)} 个专利的数据")

    # 构建图
    adj, features, node2idx, patent_list = build_graph(patent_data)

    # 训练模型
    model = train_wgcn(adj, features, num_relations=4)

    # 获取所有节点的嵌入
    model.eval()
    with torch.no_grad():
        all_embeddings = model(features, adj)

    # 提取专利节点的嵌入
    patent_indices = [node2idx[patent] for patent in patent_list]
    patent_embeddings = all_embeddings[patent_indices]

    print(f"专利嵌入矩阵形状: {patent_embeddings.shape}")  # 应该是 (1308, 32)

    # 保存专利嵌入
    save_patent_embeddings(patent_list, patent_embeddings, 'patent_embeddings.txt')

    # 如果需要单独保存每个专利的嵌入
    for i, (patent, embedding) in enumerate(zip(patent_list, patent_embeddings)):
        torch.save(embedding, f'patent_embeddings/patent_{i}.pt')
        if i >= 3:
            break


if __name__ == '__main__':
    main()
