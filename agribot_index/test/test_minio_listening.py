from minio import Minio
from dotenv import load_dotenv
import os
from pprint import pprint

load_dotenv()

'''
测试并观察 MinIO 的事件监听机制，实时显示 "test" 桶中的文件变化事件。
1. python test/test_minio_listening.py 
2. 转发端口9001
3. 在网页中向test桶中上传一个文档
4.然后观察shell命令行中输出一下内容
{'Records': [{'awsRegion': '',
              'eventName': 's3:ObjectCreated:Put',
              'eventSource': 'minio:s3',
              'eventTime': '2026-01-11T06:36:57.888Z',
              'eventVersion': '2.0',
              'requestParameters': {'principalId': 'minioadmin',
                                    'region': '',
                                    'sourceIPAddress': '172.18.0.1'},
              'responseElements': {'x-amz-id-2': 'dd9025bab4ad464b049177c95eb6ebf374d3b3fd1af9251148b658df7ac2e3e8',
                                   'x-amz-request-id': '188999C91C31D9E0',
                                   'x-minio-deployment-id': 'cf9a85c4-2d88-477b-ac2b-c80ef0e9e52e',
                                   'x-minio-origin-endpoint': 'http://172.18.0.2:9000'},
              's3': {'bucket': {'arn': 'arn:aws:s3:::test',
                                'name': 'test',
                                'ownerIdentity': {'principalId': 'minioadmin'}},
                     'configurationId': 'Config',
                     'object': {'contentType': 'application/pdf',
                                'eTag': '66139966ee80c31241de01fa7bf25c62',
                                'key': '财务管理文档.pdf',
                                'sequencer': '188999C9238B54BB',
                                'size': 386157,
                                'userMetadata': {'content-type': 'application/pdf'}},
                     's3SchemaVersion': '1.0'},
              'source': {'host': '172.18.0.1',
                         'port': '',
                         'userAgent': 'MinIO (linux; amd64) minio-go/v7.0.70 '
                                      'MinIO Console/(dev)'},
              'userIdentity': {'principalId': 'minioadmin'}}]}

'''

client = Minio(
    endpoint=os.getenv("ENDPOINT"),
    access_key=os.getenv("ACCESS_KEY"),
    secret_key=os.getenv("SECRET_KEY"),
    secure=False
)

with client.listen_bucket_notification(
    "test",
    events=["s3:ObjectCreated:*", "s3:ObjectRemoved:*"],
) as events:
    for event in events:
        pprint(event)        


