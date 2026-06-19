# Agribot

中药材种植领域的企业级 RAG（检索增强生成）问答系统。支持文档上传、向量索引、混合检索与 Web 对话界面。

## 功能概览

- **文档索引**（`agribot_index`）：监听 MinIO 上传事件，解析 PDF/Office/Markdown，写入 Milvus 向量库
- **智能问答**（`agribot_chat`）：基于 Milvus 向量检索 + 可选 Rerank + LLM 流式回答
- **管理后台**（`agribot_admin`）：浏览 MinIO 文档与 Milvus 索引状态
- **检索评测**（`experiment`）：BM25 / Dense / Hybrid / Hybrid+Rerank 评估脚本

## 项目结构

```
.
├── agribot_milvus/     # Docker：MinIO + Milvus + MySQL + MongoDB
├── agribot_index/      # 文档解析与向量入库
├── agribot_chat/       # RAG 问答 Web 服务
├── agribot_admin/      # 管理后台
└── experiment/         # RAG 检索效果评测
```

## 快速开始

### 1. 启动基础设施

```bash
cd agribot_milvus/milvus_local
cp .env.example .env          # 按需修改端口与密码
docker compose up -d
```

服务端口见 [agribot_milvus/milvus_local/readme.md](agribot_milvus/milvus_local/readme.md)。

### 2. 配置各模块环境变量

```bash
cp agribot_index/.env.example agribot_index/.env
cp agribot_chat/.env.example  agribot_chat/.env
cp agribot_admin/.env.example agribot_admin/.env
```

在 `.env` 中填写：

- MinIO / Milvus 连接信息
- LLM / Embedding / Rerank API Key（兼容 OpenAI 接口，如硅基流动）
- MySQL / MongoDB（用户登录与会话）

> **切勿将 `.env` 提交到 Git。** 仓库仅包含 `.env.example` 模板。

### 3. 安装依赖

```bash
pip install -r agribot_index/requirements.txt
pip install -r agribot_chat/requirements.txt
pip install -r agribot_admin/requirements.txt
```

### 4. 构建向量索引

```bash
# 全量刷新（首次）
cd agribot_index && python main.py --mode full-refresh

# 或实时监听 MinIO 上传
python main.py --mode listen
```

### 5. 启动问答服务

```bash
cd agribot_chat && bash run_agribot_chat
```

浏览器访问 `http://127.0.0.1:8890`。

## 部署模式

| 目录 | 说明 |
|------|------|
| `milvus_local` | 本地一体化部署（MinIO + Milvus + MySQL + MongoDB） |
| `milvus_minio` | Milvus 连接外部 MinIO |
| `milvus_oss` | Milvus 连接阿里云 OSS |

## 技术栈

- **向量库**：Milvus 2.5
- **对象存储**：MinIO / OSS
- **RAG**：LangChain + OpenAI-compatible API
- **Web**：Flask
- **数据库**：MySQL（用户）、MongoDB（会话）

## 安全说明

- 所有密钥通过环境变量注入，不要硬编码在代码或 compose 文件中
- 生产环境请修改默认密码（MinIO、MySQL、Flask Secret Key 等）
- 若曾将 `.env` 或 IDE 配置推送到远程仓库，请**立即轮换**相关 API Key 与密码

## 开源协议

[MIT License](LICENSE)

## 贡献

欢迎提交 Issue 与 Pull Request。
