import requests

response = requests.get(
    'http://localhost:8000/api/patent_recommend/',
    params={'patent_id': 'CN100000'}
)
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")