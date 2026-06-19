## 访问说明

### MinIO

- API 端口：按 OSS 控制台配置
- 管理后台：阿里云 OSS 控制台

### Milvus

- API 端口：19530
- 管理端口：9091

## docker compose 配置说明

本 compose 使用阿里云 OSS 作为 Milvus 对象存储后端。

1. 复制 `.env.example` 为 `.env` 并填写 OSS AccessKey
2. 启动：`docker compose up -d`

### standalone 环境变量示例

```yaml
MINIO_ADDRESS: oss-cn-hangzhou-internal.aliyuncs.com
MINIO_PORT: 443
MINIO_ACCESS_KEY_ID: ${MINIO_ACCESS_KEY_ID}
MINIO_SECRET_ACCESS_KEY: ${MINIO_SECRET_ACCESS_KEY}
MINIO_USE_SSL: true
MINIO_USE_VIRTUAL_HOST: true
MINIO_BUCKET_NAME: your-bucket-name
```

## S3/OSS 虚拟主机模式（Virtual Host Style）

虚拟主机模式将存储桶名称作为域名的一部分，符合 S3 标准。

- 路径模式：`http://minio.example.com:9000/bucket-name/object`
- 虚拟主机模式：`http://bucket-name.minio.example.com:9000/object`
