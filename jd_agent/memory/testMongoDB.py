from pymongo import MongoClient

# 连接本地 MongoDB
client = MongoClient("mongodb://127.0.0.1:27017")

# 测试连接（ping 命令）
try:
    client.admin.command('ping')
    print("MongoDB 连接成功！")
except Exception as e:
    print(f"连接失败: {e}")

# 列出所有数据库
print("现有数据库:", client.list_database_names())