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