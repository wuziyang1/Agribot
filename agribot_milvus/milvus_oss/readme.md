## 访问说明

### minio

- API端口：172.31.154.203:9000
- 管理后台：172.31.154.203:9001

### milvus

- API端口：19530
- 管理端口：9091



## docker compose 配置说明

### 去除
- services:
  - minio 节点

### 增加 standalone 节点的环境变量

- services:
  - standalone:  
    - enviroment:
        - MINIO_ADDRESS: oss-cn-hangzhou-internal.aliyuncs.com
        - MINIO_PORT: 443 # 使用oss时一定要设置port，否则默认是9000
        - MINIO_ACCESS_KEY_ID: 你的ak
        - MINIO_SECRET_ACCESS_KEY: 你的sk
        - MINIO_USE_SSL: true
        - MINIO_USE_VIRTUAL_HOST: true
        - MINIO_BUCKET_NAME: agribot



## S3/OSS 虚拟主机模式（Virtual Host Style）

虚拟主机模式（Virtual Host Style）是一种重要的访问方式。与传统的路径模式（Path Style）不同，虚拟主机模式将存储桶名称作为域名的一部分，这种设计更符合现代云存储服务的访问规范。

### 两种访问模式的区别

#### 路径模式（Path Style）

格式：http://minio.example.com:9000/bucket-name/object

特点：存储桶名称出现在URL路径部分


#### 虚拟主机模式（Virtual Host Style）

格式：http://bucket-name.minio.example.com:9000/object

特点：存储桶名称作为子域名，更清晰的URL结构，符合S3标准      