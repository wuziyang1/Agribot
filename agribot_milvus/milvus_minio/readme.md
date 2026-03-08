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
      - MINIO_ADDRESS: 172.31.154.203
      - MINIO_PORT: 9000
      - MINIO_ACCESS_KEY_ID: 你的ak
      - MINIO_SECRET_ACCESS_KEY: 你的sk
      - MINIO_USE_SSL: false
      - MINIO_BUCKET_NAME: agribot



      