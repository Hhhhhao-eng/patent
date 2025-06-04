from django.shortcuts import render
from django.views import View
from django.http import JsonResponse
from .utils.model_loader import get_resources
import logging
import json
from py2neo import Graph, Node, Relationship
from django.shortcuts import redirect
from django.urls import reverse
import networkx as nx
import time
import os

# 连接 Neo4j（默认密码为 'neo4j'）
graph = Graph("bolt://localhost:7687", auth=("neo4j", "zeng200312"))

def import_to_neo4j(kg_data):
    # 清空现有数据（谨慎操作！）
    graph.delete_all()
    
    # 创建节点
    nodes = {}
    for node in kg_data["nodes"]:
        # 统一创建为Entity节点（根据实际数据结构）
        n = Node("Entity", 
                 id=node["id"], 
                 label=node["label"],
                 type=node.get("type", "entity"),  # 添加type属性
                 depth=node.get("depth", 0))       # 添加depth属性
        graph.create(n)
        nodes[node["id"]] = n
    
    # 创建关系
    for link in kg_data["links"]:
        source = nodes.get(link["source"])
        target = nodes.get(link["target"])
        
        if source and target:
            # 添加关系属性
            rel = Relationship(source, link["relation"], target,
                               depth=link.get("depth", 1))
            graph.create(rel)
        else:
            logger.warning(f"Missing node for link: {link}")


logger = logging.getLogger(__name__)


class PatentRecommend(View):
    def get(self, request):
        try:
            # 获取请求参数
            patent_pid = request.GET.get('patent_id')  # 现在接收的是专利号(pid)

            if not patent_pid:
                return JsonResponse({'error': 'Missing patent_id parameter'}, status=400)

            # 获取系统资源
            resources = get_resources()
            if not resources.get("initialized"):
                logger.error("System resources not initialized")
                return JsonResponse(
                    {"error": "System initialization failed. Please check server logs."},
                    status=500
                )

            # 获取必要资源
            kc_model = resources["kc_model"]
            sacn_model = resources["sacn_model"]
            entity2id = resources.get("entity2id", {})
            patent_data_dict = resources.get("patent_data_dict", {})
            recommend_relation = resources.get("recommend_relation", "similar_to")

            # 调试信息
            print(f"Sample patent IDs: {list(patent_data_dict.keys())[:5]}")

            # 验证专利号是否存在
            if patent_pid not in patent_data_dict:
                return JsonResponse({
                    'error': f'Patent {patent_pid} not found',
                    'available_pids': list(patent_data_dict.keys())[:5]  # 返回示例专利号
                }, status=404)

            # 获取专利标题（用于实体ID查找）
            patent_title = patent_data_dict[patent_pid][1]  # 专利标题在索引1位置
            if patent_title not in entity2id:
                return JsonResponse({'error': 'Patent title mapping not found'}, status=500)

            # 执行推荐
            # 确保使用有效的关系
            recommend_relation = "similar_to"
            if recommend_relation not in resources.get("kc_model").relation2id:
                return JsonResponse(
                    {"error": f"Relation '{recommend_relation}' not available"},
                    status=400
                )

            results = kc_model.predict_missing_links(
                model=sacn_model,
                head_entity=patent_title,
                relation=recommend_relation,
                top_k=10
            )

            # 构建推荐结果
            patent_list = []
            if results:
                for entity_title, score in results:
                    # 通过标题查找对应的专利号
                    matched_pid = next(
                        (pid for pid, data in patent_data_dict.items()
                         if data[1] == entity_title),  # data[1]是标题
                        None
                    )

                    if matched_pid and matched_pid in patent_data_dict:
                        p = patent_data_dict[matched_pid]
                        patent_list.append({
                            'id': matched_pid,  # 使用专利号作为ID
                            'pid': p[0],  # 专利号
                            'title': p[1],  # 专利标题
                            'patType': p[2],  # 专利类型
                            'appNumber': p[3],  # 申请号
                            'appDate': p[4],  # 申请日
                            'pubNumber': p[5],  # 公开号
                            'grantDate': p[6],  # 授权日
                            'mainIpc': p[7],  # 主分类号
                            'ipc': p[8],  # 分类号
                            'applicantName': p[9],  # 申请人
                            'inventorName': p[10],  # 发明人
                            'address': p[11],  # 地址
                            'abs': p[12],  # 摘要
                            'score': float(score)  # 相似度分数
                        })

            return JsonResponse({'obj': patent_list})

        except Exception as e:
            logger.error(f"API error: {str(e)}", exc_info=True)
            return JsonResponse(
                {"error": "Internal server error"},
                status=500
            )
        

class PatentDetail(View):
    """专利详情接口"""
    def get(self, request):
        try:
            patent_pid = request.GET.get('patent_id')
            
            if not patent_pid:
                return JsonResponse({'error': 'Missing patent_id parameter'}, status=400)
            
            resources = get_resources()
            patent_data_dict = resources.get("patent_data_dict", {})
            
            if patent_pid not in patent_data_dict:
                return JsonResponse({
                    'error': f'Patent {patent_pid} not found',
                    'available_pids': list(patent_data_dict.keys())[:5]
                }, status=404)
            
            p = patent_data_dict[patent_pid]
            patent_detail = {
                'id': patent_pid,  # 专利ID
                'pid': p[0],      # 专利号
                'title': p[1],     # 专利标题
                'patType': p[2],   # 专利类型
                'appNumber': p[3], # 申请号
                'appDate': p[4],   # 申请日
                'pubNumber': p[5], # 公开号
                'grantDate': p[6], # 授权日
                'mainIpc': p[7],  # 主分类号
                'ipc': p[8],       # 分类号
                'applicantName': p[9],  # 申请人
                'inventorName': p[10],  # 发明人
                'address': p[11],      # 地址
                'abs': p[12]            # 摘要
            }
            
            return JsonResponse({'data': patent_detail})
            
        except Exception as e:
            logger.error(f"Patent detail error: {str(e)}", exc_info=True)
            return JsonResponse(
                {"error": "Internal server error"},
                status=500
            )


class KnowledgeGraphData(View):
    """知识图谱数据接口"""
    def get(self, request):
        try:
            patent_id = request.GET.get('patent_id')
            entity_id = request.GET.get('entity_id')
            
            if not patent_id and not entity_id:
                return JsonResponse({'error': 'Missing patent_id or entity_id'}, status=400)
            
            resources = get_resources()
            kc_model = resources["kc_model"]
            
            # 优先使用patent_id，若无则使用entity_id
            target_id = patent_id if patent_id else entity_id
            
            # 获取知识图谱结构数据
            kg_data = kc_model.get_kg_structure(target_id,depth=1)
            
            if "error" in kg_data:
                return JsonResponse(kg_data, status=404)
                
            return JsonResponse({'data': kg_data})
            
        except Exception as e:
            logger.error(f"Knowledge graph error: {str(e)}", exc_info=True)
            return JsonResponse(
                {"error": "Internal server error"},
                status=500
            )

class Neo4jExport(View):
    """导出知识图谱数据到 Neo4j"""
    def get(self, request):
        try:
            patent_id = request.GET.get('patent_id')
            entity_id = request.GET.get('entity_id')
            
            if not patent_id and not entity_id:
                return JsonResponse({'error': 'Missing patent_id or entity_id'}, status=400)
            
            resources = get_resources()
            kc_model = resources["kc_model"]
            target_id = patent_id if patent_id else entity_id
            
            # 获取知识图谱结构数据
            kg_data = kc_model.get_kg_structure(target_id, depth=2)
            
            if "error" in kg_data:
                return JsonResponse(kg_data, status=404)
            
            # 确保数据结构正确
            if not isinstance(kg_data, dict) or "nodes" not in kg_data or "links" not in kg_data:
                return JsonResponse({"error": "Invalid knowledge graph data format"}, status=500)
            
            # 导入到 Neo4j
            import_to_neo4j(kg_data)
            
            # 重定向到 Neo4j Browser
            neo4j_url = "http://localhost:7474/browser/"
            return redirect(neo4j_url)
        except Exception as e:
            logger.error(f"Neo4j export error: {str(e)}", exc_info=True)
            return JsonResponse({"error": str(e)}, status=500)

       

class KGPCInferRelation(View):
    """实体关系推理接口"""
    def get(self, request):
        try:
            entity1 = request.GET.get('entity1')
            entity2 = request.GET.get('entity2')
            top_k = int(request.GET.get('top_k', 5))  # 默认返回前5个关系
            
            if not entity1 or not entity2:
                return JsonResponse({'error': 'Missing entity1 or entity2 parameter'}, status=400)
            
            resources = get_resources()
            kc_model = resources["kc_model"]
            sacn_model = resources["sacn_model"]
            entity2id = resources.get("entity2id", {})
            id2entity = resources.get("id2entity", {})

            # 验证实体是否存在
            if entity1 not in entity2id:
                entity1 = id2entity[int(entity1[2:])-100000]
                if entity1 not in entity2id:
                    return JsonResponse({'error': f'Entity "{entity1}" not found'}, status=404)
            if entity2 not in entity2id:
                entity2 = id2entity[int(entity2[2:])-100000]
                if entity1 not in entity2id:
                    return JsonResponse({'error': f'Entity "{entity2}" not found'}, status=404)
            
            # 预测两个实体之间可能的关系
            start_time = time.time()
            predicted_relations = kc_model.predict_relation(
                model=sacn_model,
                head_entity=entity1,
                tail_entity=entity2,
                top_k=top_k
            )
            inference_time = time.time() - start_time
            
            # 格式化结果
            results = []
            for relation, score in predicted_relations:
                results.append({
                    'relation': relation,
                    'score': float(score),
                    'explanation': f"{entity1} 与 {entity2} 可能存在 {relation} 关系"
                })
            
            return JsonResponse({
                'status': 'success',
                'entity1': entity1,
                'entity2': entity2,
                'results': results,
                'inference_time': f"{inference_time:.3f}秒"
            })
            
        except Exception as e:
            logger.error(f"Relation inference error: {str(e)}", exc_info=True)
            return JsonResponse(
                {"error": "Internal server error"},
                status=500
            )


class KGPCFindPath(View):
    """实体间路径发现接口（结合Neo4j和模型预测）"""
    def get(self, request):
        try:
            entity1 = request.GET.get('entity1')
            entity2 = request.GET.get('entity2')
            max_length = int(request.GET.get('max_length', 3))  # 默认最大路径长度3
            top_k = int(request.GET.get('top_k', 5))  # 默认返回前5条路径
            
            if not entity1 or not entity2:
                return JsonResponse({'error': 'Missing entity1 or entity2 parameter'}, status=400)
            
            resources = get_resources()
            kc_model = resources["kc_model"]
            sacn_model = resources["sacn_model"]
            entity2id = resources.get("entity2id", {})
            id2entity = resources.get("id2entity", {})
            
            # 验证实体是否存在
            if entity1 not in entity2id:
                # 尝试转换专利ID格式
                try:
                    entity1 = id2entity[int(entity1[2:])-100000]
                except:
                    return JsonResponse({'error': f'Entity "{entity1}" not found'}, status=404)
            if entity2 not in entity2id:
                try:
                    entity2 = id2entity[int(entity2[2:])-100000]
                except:
                    return JsonResponse({'error': f'Entity "{entity2}" not found'}, status=404)
            
            start_time = time.time()
            paths = []
            
            # 1. 首先尝试从Neo4j中查找路径
            try:
                # 查询节点
                nodes_query = """
                MATCH (n) 
                RETURN n.id AS id, labels(n)[0] AS label, n.type AS type
                """
                nodes_data = graph.run(nodes_query).data()
                
                # 查询关系
                links_query = """
                MATCH (s)-[r]->(t) 
                RETURN s.id AS source, t.id AS target, type(r) AS relation
                """
                links_data = graph.run(links_query).data()
                
                # 构建图
                G = nx.MultiDiGraph()
                
                # 添加节点
                for node in nodes_data:
                    G.add_node(node['id'], label=node['label'], type=node.get('type', 'entity'))
                
                # 添加边
                for link in links_data:
                    G.add_edge(
                        link['source'], 
                        link['target'], 
                        relation=link['relation']
                    )
                
                # 查找路径
                try:
                    # 使用BFS查找所有简单路径
                    for path in nx.all_simple_edge_paths(G, source=entity1, target=entity2, cutoff=max_length):
                        if len(path) <= max_length:
                            path_info = []
                            for u, v, key in path:
                                edge_data = G.get_edge_data(u, v, key)
                                path_info.append({
                                    'source': u,
                                    'target': v,
                                    'relation': edge_data['relation'],
                                    'source_type': 'neo4j'  # 标记来源
                                })
                            paths.append(path_info)
                            
                            # 限制返回数量
                            if len(paths) >= top_k:
                                break
                except nx.NetworkXNoPath:
                    pass  # 没有找到路径
            except Exception as e:
                logger.warning(f"Neo4j path finding failed: {str(e)}")
            
            # 2. 如果Neo4j中没有找到路径，则使用模型预测
            if not paths:
                # 使用模型预测双向路径
                start_time = time.time()
                
                # 1. 尝试正向路径预测 (entity1 -> entity2)
                forward_paths = kc_model.predict_path(
                    model=sacn_model,
                    head_entity=entity1,
                    tail_entity=entity2,
                    max_depth=max_length,
                    beam_size=top_k,
                    threshold=0.5
                )
                
                # 2. 尝试反向路径预测 (entity2 -> entity1)
                reverse_paths = kc_model.predict_path(
                    model=sacn_model,
                    head_entity=entity2,
                    tail_entity=entity1,
                    max_depth=max_length,
                    beam_size=top_k,
                    threshold=0.5
                )
                
                # 3. 合并并排序所有路径
                all_paths = forward_paths + reverse_paths
                all_paths.sort(key=lambda x: x[1], reverse=True)  # 按概率降序排序
                
                # 取前top_k条路径
                selected_paths = all_paths[:top_k]
                
                inference_time = time.time() - start_time
                
                # 格式化模型预测的路径
                paths = []
                for path, prob in selected_paths:
                    path_info = []
                    # 路径格式: [e1, r1, e2, r2, e3, ...]
                    # 遍历路径中的每个关系步骤
                    for i in range(0, len(path)-1, 2):
                        if i+2 >= len(path):
                            break
                        source = path[i]
                        relation = path[i+1]
                        target = path[i+2]
                        
                        # 检查是否是反向路径
                        is_reverse = (source == entity2 and target == entity1)
                        
                        path_info.append({
                            'source': source,
                            'target': target,
                            'relation': relation,
                            'probability': float(prob),
                            'is_reverse': is_reverse
                        })
                    paths.append(path_info)
            
            inference_time = time.time() - start_time
            
            return JsonResponse({
                'status': 'success',
                'entity1': entity1,
                'entity2': entity2,
                'paths': paths,
                'path_count': len(paths),
                'inference_time': f"{inference_time:.3f}秒"
            })
            
        except Exception as e:
            logger.error(f"Path finding error: {str(e)}", exc_info=True)
            return JsonResponse(
                {"error": "Internal server error"},
                status=500
            )

class EntityListView(View):
    """返回实体名列表（前5000个）"""
    def get(self, request):
        try:
            resources = get_resources()
            entity2id = resources.get("entity2id", {})
            entity_list = list(entity2id.keys())[:5000]
            return JsonResponse({'entities': entity_list})
        except Exception as e:
            logger.error(f"Entity list error: {str(e)}", exc_info=True)
            return JsonResponse({"error": "Internal server error"}, status=500)

# 工具函数：导出所有pid-title对应关系到txt文件
def export_pid_title_map():
    resources = get_resources()
    patent_data_dict = resources.get("patent_data_dict", {})
    out_path = os.path.join(os.path.dirname(__file__), "pid_title_map.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for pid, data in patent_data_dict.items():
            f.write(f"{pid}\t{data[1]}\n")
    print(f"导出完成: {out_path}")

# 你可以在Django shell中调用：
# from kgpc.views import export_pid_title_map
# export_pid_title_map()
# 或临时在某个接口调用它
