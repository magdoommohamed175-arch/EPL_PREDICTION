import requests

url = "https://fbref.com/en/comps/9/Premier-League-Stats"

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=headers)

print("Status:", response.status_code)
print(response.text[:500])