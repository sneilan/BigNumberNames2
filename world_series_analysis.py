import pandas as pd
import requests
from io import StringIO

# Fetch World Series data from Wikipedia with proper headers
url = "https://en.wikipedia.org/wiki/List_of_World_Series_champions"

print("Fetching World Series data from Wikipedia...")

# Set up headers to avoid 403 error
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Fetch the page
response = requests.get(url, headers=headers)
response.raise_for_status()

# Read tables from the HTML
tables = pd.read_html(StringIO(response.text))

# The first table should contain World Series winners
print(f"\nFound {len(tables)} tables on the page")

# Let's examine the first few tables to find the right one
for i, table in enumerate(tables[:5]):
    print(f"\n--- Table {i} ---")
    print(f"Shape: {table.shape}")
    print(f"Columns: {list(table.columns)}")
    if len(table) > 0:
        print(f"First row sample: {table.iloc[0].to_dict()}")
