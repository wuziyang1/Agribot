## 访问说明

### MinIO（外部实例）

- API 端口：`127.0.0.1:9000`（按实际部署修改）
- 管理后台：`127.0.0.1:9001`

### Milvus

- API 端口：19530
- 管理端口：9091

## docker compose 配置说明

本 compose 不包含 MinIO 容器，Milvus 连接**外部 MinIO**。

1. 复制 `.env.example` 为 `.env` 并填写 MinIO 密钥
2. 启动：`docker compose up -d`

### standalone 环境变量示例

```yaml
MINIO_ADDRESS: ${MINIO_ADDRESS}
MINIO_PORT: ${MINIO_PORT}
MINIO_ACCESS_KEY_ID: ${MINIO_ACCESS_KEY_ID}
MINIO_SECRET_ACCESS_KEY: ${MINIO_SECRET_ACCESS_KEY}
MINIO_USE_SSL: ${MINIO_USE_SSL}
MINIO_BUCKET_NAME: ${MINIO_BUCKET_NAME}
```
