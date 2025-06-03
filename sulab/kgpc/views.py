from django.shortcuts import render
from django.views import View
from django.http import JsonResponse
from .utils.model_loader import get_resources
import logging

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
            patent_pid = request.GET.get('patent_id')
            entity_id = request.GET.get('entity_id')
            
            if not patent_pid and not entity_id:
                return JsonResponse({'error': 'Missing patent_id or entity_id parameter'}, status=400)
            
            resources = get_resources()
            patent_data_dict = resources.get("patent_data_dict", {})
            entity2id = resources.get("entity2id", {})
            kc_model = resources["kc_model"]
            
            # 这里简化处理，实际应根据您的知识图谱结构返回相关数据
            if patent_pid:
                if patent_pid not in patent_data_dict:
                    return JsonResponse({'error': 'Patent not found'}, status=404)
                
                patent_title = patent_data_dict[patent_pid][1]
                if patent_title not in entity2id:
                    return JsonResponse({'error': 'Patent title mapping not found'}, status=500)
                
                # 获取与专利相关的实体和关系
                # 这里需要根据您的知识图谱结构实现具体逻辑
                graph_data = {
                    'nodes': [
                        {'id': patent_title, 'type': 'patent', 'label': patent_title}
                    ],
                    'links': []
                }
                
                # 示例：添加一些相关实体（需要根据实际数据实现）
                # 这里应该从知识图谱中查询与当前专利相关的实体和关系
                
            elif entity_id:
                # 处理实体查询
                if entity_id not in entity2id:
                    return JsonResponse({'error': 'Entity not found'}, status=404)
                
                graph_data = {
                    'nodes': [
                        {'id': entity_id, 'type': 'entity', 'label': entity_id}
                    ],
                    'links': []
                }
            
            return JsonResponse({'data': graph_data})
            
        except Exception as e:
            logger.error(f"Knowledge graph error: {str(e)}", exc_info=True)
            return JsonResponse(
                {"error": "Internal server error"},
                status=500
            )


class KGPCInference(View):
    """AI/知识图谱推理接口"""
    def get(self, request):
        try:
            inference_type = request.GET.get('type')
            entity1 = request.GET.get('entity1')
            entity2 = request.GET.get('entity2')
            relation = request.GET.get('relation')
            
            resources = get_resources()
            kc_model = resources["kc_model"]
            sacn_model = resources["sacn_model"]
            entity2id = resources.get("entity2id", {})
            
            if not inference_type:
                return JsonResponse({'error': 'Missing type parameter'}, status=400)
            
            if inference_type == 'relation' and entity1 and entity2:
                # 关系推理
                if entity1 not in entity2id or entity2 not in entity2id:
                    return JsonResponse({'error': 'Entity not found'}, status=404)
                
                # 预测两个实体之间可能的关系
                predicted_relations = kc_model.predict_relation(
                    model=sacn_model,
                    head_entity=entity1,
                    tail_entity=entity2,
                    top_k=5
                )
                
                return JsonResponse({
                    'data': {
                        'entity1': entity1,
                        'entity2': entity2,
                        'possible_relations': predicted_relations
                    }
                })
                
            elif inference_type == 'path' and entity1 and entity2:
                # 路径发现
                if entity1 not in entity2id or entity2 not in entity2id:
                    return JsonResponse({'error': 'Entity not found'}, status=404)
                
                # 查找两个实体之间的路径
                # 这里需要根据您的知识图谱结构实现具体逻辑
                paths = []
                # paths = kc_model.find_paths(entity1, entity2, max_length=3)
                
                return JsonResponse({
                    'data': {
                        'entity1': entity1,
                        'entity2': entity2,
                        'paths': paths
                    }
                })
                
            else:
                return JsonResponse({'error': 'Invalid parameters for inference type'}, status=400)
            
        except Exception as e:
            logger.error(f"Inference error: {str(e)}", exc_info=True)
            return JsonResponse(
                {"error": "Internal server error"},
                status=500
            )