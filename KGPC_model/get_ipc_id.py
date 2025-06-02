import requests
import json
import time

# 配置API参数
API_URL = "https://api.cnipa.gov.cn/advancedSearch"  # CNIPA示例URL（实际URL需参考文档）
API_KEY = "YOUR_API_KEY"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

def get_ipc_by_title(title):
    """通过专利标题查询IPC分类号"""
    params = {
        "keyword": f'TI="{title}"',  # 按标题精确检索
        "fields": "ipc"              # 仅返回IPC字段
    }
    try:
        response = requests.get(API_URL, headers=HEADERS, params=params, timeout=10)
        data = response.json()
        
        # 解析IPC（示例：取第一条专利的主IPC）
        if data.get("data") and data["data"]["total"] > 0:
            first_patent = data["data"]["list"][0]
            return first_patent.get("ipc", "N/A").split(";")[0]  # 取第一个IPC分类
        return "N/A"
    except Exception as e:
        print(f"查询失败: {title}, 错误: {str(e)}")
        return "N/A"

# 读取专利标题文件
with open("patent_titles.txt", "r", encoding="utf-8") as f:
    titles = [line.strip() for line in f.readlines()]

# 批量查询并写入结果
with open("patent_ipc_mapping.txt", "w", encoding="utf-8") as out_file:
    for idx, title in enumerate(titles):
        ipc = get_ipc_by_title(title)
        out_file.write(f"{title}---{ipc}\n")
        print(f"进度: {idx+1}/{len(titles)} - {title} -> {ipc}")
        time.sleep(1)  # 避免频繁请求（根据API限流调整）