import tomllib
from pymongo import MongoClient

uri = tomllib.load(open(".streamlit/secrets.toml", "rb"))["MONGODB_URI"]
try:
    MongoClient(uri, serverSelectionTimeoutMS=8000).admin.command("ping")
    print("OK, terhubung")
except Exception as e:
    print(type(e).__name__)
    print(str(e)[:700].replace(uri, "<uri>"))