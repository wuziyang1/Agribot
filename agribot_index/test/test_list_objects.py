from minio import Minio
from dotenv import load_dotenv
import os
from pprint import pprint

load_dotenv()

'''
连接到 MinIO 服务器
列出 "test" 桶中的所有文件（对象）
显示每个文件的详细信息
统计文件总数
'''
client = Minio(
    endpoint=os.getenv("ENDPOINT"),
    access_key=os.getenv("ACCESS_KEY"),
    secret_key=os.getenv("SECRET_KEY"),
    secure=False
)

objs = client.list_objects("test", recursive=True)
count = 0
for obj in objs:
    pprint("--------------------------------")
    pprint(f"object_name: {obj.object_name}")
    pprint(f"last_modified: {obj.last_modified}")
    pprint(f"size: {obj.size}")
    pprint(f"etag: {obj.etag}")
    pprint(f"content_type: {obj.content_type}")
    pprint(f"tags: {obj.tags}")
    pprint(f"storage_class: {obj.storage_class}")
    count += 1

pprint("--------------------------------")
pprint(f"count: {count}")





