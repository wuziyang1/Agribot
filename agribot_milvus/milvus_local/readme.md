## Agribot 全部数据库 - 一键启动

### 启动 / 停止

```bash
# 启动全部服务
docker compose up -d

# 停止全部服务
docker compose down

# 停止并删除数据卷（⚠️ 会清空所有数据）
docker compose down -v
```

### 服务端口说明

| 服务 | 端口 | 说明 |
|------|-----|------|
| MinIO | 9000 | S3 兼容 API |
| MinIO | 9001 | 管理后台 |
| Milvus | 19530 | gRPC API |
| Milvus | 9091 | 管理/健康检查 |
| MongoDB | 27017 | 默认端口 |
| MySQL | 3306 | 默认端口 |
| Neo4j | 7474 | HTTP / 浏览器 |
| Neo4j | 7687 | Bolt 协议 |

### 默认账号

| 服务 | 用户名 | 密码 |
|------|--------|------|
| MinIO | minioadmin | minioadmin |
| MongoDB | agribot | agribot123 |
| MySQL | agribot | agribot123 |
| MySQL (root) | root | agribot_root123 |
| Neo4j | neo4j | REDACTED_NEO4J_PASSWORD |

> 所有配置均可在 `.env` 文件中修改。数据持久化到 `DOCKER_VOLUME_DIRECTORY` 指定的目录。
