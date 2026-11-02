import os
import requests
import pandas as pd

TOKEN = os.getenv("FOOTBALL_DATA_TOKEN")

url = "https://api.football-data.org/v4/competitions/PL/scorers"

headers = {
    "X-Auth-Token": TOKEN
}

response = requests.get(url, headers=headers)

print("Status:", response.status_code)

data = response.json()

rows = []

for item in data["scorers"]:
    rows.append({
        "player": item["player"]["name"],
        "team": item["team"]["name"],
        "goals": item["goals"],
        "assists": item["assists"],
        "played_matches": item["playedMatches"]
    })

df = pd.DataFrame(rows)

print("\nPlayer Data:")
print(df)

print("\nNumber of players:", len(df))