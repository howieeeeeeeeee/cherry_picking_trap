import os
from pymongo import MongoClient


MONGOCLIENT_LOCAL = MongoClient(
    "mongodb://localhost:27017",
    serverSelectionTimeoutMS=30000,  # 30 seconds
    socketTimeoutMS=30000,  # 30 seconds
    connectTimeoutMS=30000,  # 30 seconds
    maxPoolSize=50,
    retryWrites=True,
    retryReads=True,
)


MONGOCLIENT = MongoClient(
    "mongodb+srv://cluster0.ni3d51g.mongodb.net",
    username=os.environ.get("MONGO_USERNAME"),
    password=os.environ.get("MONGO_PASSWORD"),
    serverSelectionTimeoutMS=30000,  # 30 seconds
    socketTimeoutMS=30000,  # 30 seconds
    connectTimeoutMS=30000,  # 30 seconds
    maxPoolSize=50,
    retryWrites=True,
    retryReads=True,
)
