# 系统 (MiDoc RAG Platform)

> 一个基于 **MinIO + Milvus + LangChain** 的企业级文档知识库 RAG 问答系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 项目简介

本系统是一个**端到端的企业文档知识库 RAG 问答系统**，旨在帮助企业将内部文档（PDF、Word、Excel、Markdown 等）转化为可搜索、可问答的智能知识库。

### 核心价值

- 🚀 **自动化文档索引**：支持多种文档格式，自动解析、分块、向量化
- 🔍 **智能语义检索**：基于 Milvus 向量数据库的高性能相似度搜索
- 💬 **RAG 增强生成**：结合向量检索与大语言模型，生成基于知识库的准确回答
- 🎯 **企业级特性**：支持增量更新、重排序优化、多模型切换
- 🛠️ **可视化管理**：提供 Web 管理后台，方便文档管理和索引监控

---

## 🏗️ 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                     用户交互层                                │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │  Web 问答页  │  │  API 接口    │                        │
│  └──────────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    RAG 服务层                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  RAGService (LangChain + Milvus)                     │   │
│  │  • 向量检索 (相似度搜索)                              │   │
│  │  • 重排序优化 (可选)                                  │   │
│  │  • LLM 生成回答                                       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    数据存储层                                │
│  ┌──────────────┐              ┌──────────────┐           │
│  │   MinIO      │              │   Milvus      │           │
│  │  (文档存储)   │              │  (向量数据库) │           │
│  └──────────────┘              └──────────────┘           │
└─────────────────────────────────────────────────────────────┘
                            ↑
┌─────────────────────────────────────────────────────────────┐
│                    索引服务层                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  MildocIndex (文档解析 + Embedding + 索引)            │   │
│  │  • 文档解析 (PDF/Word/Excel/Markdown/Text)           │   │
│  │  • 文本分块 (LangChain TextSplitter)                 │   │
│  │  • 向量生成 (OpenAI Embedding API)                    │   │
│  │  • 增量更新 (MinIO 事件监听)                          │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 数据流程

1. **文档入库流程**：
   ```
   文档上传到 MinIO → MinIO 事件触发 → 文档解析 → 文本分块 → 
   Embedding 向量化 → 存储到 Milvus → 索引完成
   ```

2. **问答流程**：
   ```
   用户问题 → Embedding 向量化 → Milvus 相似度检索 → 
   重排序优化（可选）→ LLM 生成回答 → 返回结果
   ```

---

## 📦 项目结构

```
rag/
├── mildoc_index/          # 文档索引服务
│   ├── main.py           # 主程序（支持全量/增量/监听模式）
│   ├── embedding.py      # Embedding 工具类
│   ├── milvus_api.py     # Milvus 数据库操作
│   ├── parser/           # 文档解析器
│   │   ├── simple_object_parser.py  # 统一解析入口
│   │   ├── pdf_parser.py             # PDF 解析器
│   │   ├── office_parser.py          # Office 文档解析器
│   │   ├── markdown_parser.py        # Markdown 解析器
│   │   └── text_parser.py            # 纯文本解析器
│   └── requirements.txt
│
├── mildoc_wxkf/          # RAG 服务
│   ├── rag_service.py    # RAG 核心服务（LangChain + Milvus）
│   ├── rerank_service.py # 重排序服务（支持阿里百炼/硅基流动）
│   ├── config.py         # 配置管理
│   └── requirements.txt
│
├── mildoc_admin/         # Web 管理后台
│   ├── admin_app.py      # Flask 管理应用
│   ├── templates/        # HTML 模板
│   ├── static/           # 静态资源
│   └── requirements.txt
│
└── mildoc_milvus/        # Milvus 部署相关
    ├── milvus_local/     # 本地部署配置
    ├── milvus_minio/     # MinIO 存储后端配置
    └── milvus_oss/       # OSS 存储后端配置
```

---

## ✨ 核心功能

### 1. 文档索引服务 (`mildoc_index`)

#### 支持的文档格式
- ✅ **PDF**：使用 PyPDF2 解析
- ✅ **Office 文档**：Word (.docx)、Excel (.xlsx)、PowerPoint (.pptx) - 使用 markitdown
- ✅ **Markdown**：`.md` 文件
- ✅ **纯文本**：`.txt` 文件
- 🔄 **扩展性**：支持自定义解析器

#### 索引模式

**模式 1：全量刷新** (`--mode full-refresh`)
- 遍历 MinIO 桶中所有文档
- 删除旧索引，重新解析和向量化
- 适用于首次构建或重建索引

**模式 2：排查补漏** (`--mode backfill`)
- 检查 MinIO 中未索引的文档
- 只处理缺失的文档，跳过已存在的
- 适用于日常维护和补漏

**模式 3：增量更新** (`--mode listen`)
- 实时监听 MinIO 事件（文件上传/删除）
- 自动处理新增和删除的文档
- 适用于生产环境持续更新

#### 文本分块策略
- **分块大小**：默认 2048 字符
- **重叠大小**：默认 128 字符（保证上下文连续性）
- **分块算法**：LangChain `RecursiveCharacterTextSplitter`

### 2. RAG 服务 (`mildoc_wxkf`)

#### 核心能力

**向量检索**
- 基于 Milvus 的相似度搜索（余弦相似度）
- 支持 IVF_FLAT 索引（nlist=1024, nprobe=64）
- 初始检索 Top-K（默认 10 个候选文档）

**重排序优化**（可选）
- 支持多个重排序服务提供商：
  - 阿里百炼平台 (`dashscope`)
  - 硅基流动平台 (`siliconflow`)
- 对初始检索结果进行相关性重排序
- 提高答案准确性

**LLM 生成**
- 基于 LangChain 的 RAG 流程
- 支持 OpenAI 兼容的 API（可配置）
- 专业客服风格的提示词模板
- Token 使用情况追踪

#### RAG 流程

```python
用户问题
  ↓
1. Embedding 向量化
  ↓
2. Milvus 相似度检索（Top-10）
  ↓
3. 重排序优化（Top-5，可选）
  ↓
4. 构建上下文 Prompt
  ↓
5. LLM 生成回答
  ↓
返回结果（答案 + 参考文档 + Token 统计）
```

### 3. Web 管理后台 (`mildoc_admin`)

#### 功能特性

- 📁 **文件浏览**：类似文件管理器的 MinIO 文档浏览
- 📄 **文档详情**：查看文档在 MinIO 和 Milvus 中的状态
- 🔍 **索引监控**：查看文档的索引状态、分块数量、MD5 校验
- 📤 **文件上传**：支持批量上传文档到 MinIO
- 🗑️ **文件删除**：删除 MinIO 文件和对应的 Milvus 索引
- 📂 **目录管理**：创建/删除目录结构

---

## 🛠️ 技术栈

### 后端技术

| 技术 | 版本 | 用途 |
|------|------|------|
| **Python** | 3.8+ | 开发语言 |
| **LangChain** | 0.3.x | RAG 框架 |
| **Milvus** | 2.5+ | 向量数据库 |
| **MinIO** | - | 对象存储 |
| **Flask** | 3.1+ | Web 框架 |
| **OpenAI API** | 1.0+ | Embedding & LLM |
| **PyPDF2** | 3.0+ | PDF 解析 |
| **markitdown** | 0.1+ | Office 文档解析 |

### 核心依赖

```python
# 向量数据库
pymilvus==2.5.15
langchain-milvus==0.2.1

# RAG 框架
langchain==0.3.26
langchain-openai==0.3.28
langchain-text-splitters==0.3.9

# 文档解析
pypdf2==3.0.1
markitdown[pdf,docx,pptx,xlsx]==0.1.2

# 对象存储
minio==7.2.16

# Web 框架
Flask==3.1.1
```

---

## 🚀 快速开始

### 前置要求

1. **Python 3.8+**
2. **Milvus 2.5+**（向量数据库）
3. **MinIO**（对象存储）
4. **OpenAI API Key**（或兼容的 Embedding/LLM API）

### 安装步骤

#### 1. 克隆项目

```bash
git clone <repository-url>
cd rag
```

#### 2. 配置环境变量

创建 `.env` 文件，配置以下变量：

```bash
# MinIO 配置
ENDPOINT=localhost:9000
ACCESS_KEY=minioadmin
SECRET_KEY=minioadmin
MINIO_BUCKET=public-docs

# Milvus 配置
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_USER=root
MILVUS_PASSWORD=Milvus
MILVUS_DATABASE=default
MILVUS_COLLECTION=mildoc_collection
MILVUS_INDEX_NAME=content_vector_index
MILVUS_VECTOR_DIM=1536
MILVUS_INDEX_TYPE=IVF_FLAT

# OpenAI API 配置（Embedding）
LLM_EMBEDDING_MODEL_NAME=text-embedding-3-small
LLM_EMBEDDING_API_KEY=your-api-key
LLM_EMBEDDING_BASE_URL=https://api.openai.com/v1

# LLM 配置（生成回答）
LLM_MODEL_NAME=gpt-3.5-turbo
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.openai.com/v1

# 重排序服务（可选）
RERANK_PROVIDER=dashscope  # 或 siliconflow
RERANK_API_KEY=your-rerank-api-key
RERANK_MODEL_NAME=gte-rerank-v2
RERANK_ENDPOINT=https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/rerank
```

#### 3. 安装依赖

```bash
# 安装索引服务依赖
cd mildoc_index
pip install -r requirements.txt

# 安装 RAG 服务依赖
cd ../mildoc_wxkf
pip install -r requirements.txt

# 安装管理后台依赖
cd ../mildoc_admin
pip install -r requirements.txt
```

#### 4. 初始化索引

```bash
cd mildoc_index

# 全量刷新模式（首次运行）
python main.py --mode full-refresh

# 或增量监听模式（生产环境）
python main.py --mode listen
```

#### 5. 启动 RAG 服务

```bash
cd mildoc_wxkf
# 启动 Web API 服务（需要创建 Web API 接口）
```

#### 6. 启动管理后台

```bash
cd mildoc_admin
python admin_app.py
```

访问 `http://localhost:5000` 登录管理后台。

---

## 📝 使用示例

### 索引文档

```bash
# 1. 上传文档到 MinIO（通过管理后台或 MinIO 客户端）

# 2. 全量刷新索引
python mildoc_index/main.py --mode full-refresh

# 3. 或启动增量监听（自动处理新文档）
python mildoc_index/main.py --mode listen
```

### 使用 RAG 服务

```python
from mildoc_wxkf.rag_service import get_rag_service

# 获取 RAG 服务实例
rag_service = get_rag_service()

# 查询问题
response = rag_service.query_service("什么是 RAG？", use_rerank=True)

# 查看结果
print(f"回答：{response.content}")
print(f"参考文档：{[doc.doc_name for doc in response.source_documents]}")
print(f"Token 使用：{response.token_usage.total_tokens}")
```

### 管理后台操作

1. **浏览文档**：访问 `/files` 查看 MinIO 中的文档
2. **查看详情**：点击文档查看索引状态和分块信息
3. **上传文档**：通过上传功能添加新文档
4. **删除文档**：删除文档时会自动清理 Milvus 索引

---

## 🔧 配置说明

### Milvus 索引配置

```python
# 索引类型：IVF_FLAT
# 参数：
#   - nlist: 1024（聚类中心数量）
#   - nprobe: 64（查询时扫描的聚类数量）
# 相似度度量：余弦相似度 (COSINE)
```

### 文本分块配置

```python
# 默认配置
chunk_size = 2048      # 每个分块最大字符数
chunk_overlap = 128    # 分块之间的重叠字符数
```

### RAG 检索配置

```python
# 初始检索
initial_k = 10  # 如果启用重排序，检索 10 个候选文档

# 重排序后
final_k = 3     # 最终使用前 3 个文档生成回答
```

---

## 🎯 核心特性详解

### 1. 增量更新机制

- **MinIO 事件监听**：实时监听文件上传/删除事件
- **消息去重**：内存缓存 + 数据库双重检查，避免重复处理
- **Cursor 机制**：持久化保存拉取位置，避免消息丢失
- **时效性检查**：只处理 10 分钟内的消息

### 2. 重排序优化

- **多提供商支持**：阿里百炼、硅基流动
- **相关性评分**：基于查询和文档的相关性重排序
- **安全检查**：确保原始最高相似度文档不会被过滤掉

### 3. 文档解析策略

- **多解析器支持**：按优先级尝试不同解析器
- **文件大小限制**：默认 512MB，超过则跳过
- **MD5 校验**：用于文档去重和变更检测

### 4. 错误处理与日志

- **完善的日志系统**：记录每个步骤的详细信息
- **异常处理**：优雅处理各种异常情况
- **健康检查**：提供组件健康状态检查接口

---

## 📊 性能指标

### 索引性能

- **文档解析速度**：取决于文档大小和类型
- **向量化速度**：受 Embedding API 限制
- **批量插入**：支持批量插入到 Milvus

### 检索性能

- **向量检索**：毫秒级响应（取决于数据量）
- **重排序**：增加 100-500ms 延迟（取决于提供商）
- **LLM 生成**：1-5 秒（取决于模型和回答长度）

---

## 🔐 安全特性

- **环境变量配置**：敏感信息通过环境变量管理
- **访问控制**：管理后台支持登录验证
- **数据校验**：MD5 校验确保文档完整性
- **错误处理**：避免敏感信息泄露

---

## 🚧 未来规划

- [ ] **Web 问答界面**：提供完整的 Web 问答页面
- [ ] **多租户支持**：支持多个知识库隔离
- [ ] **API 接口**：提供 RESTful API 供第三方集成
- [ ] **流式回答**：支持流式输出，提升用户体验
- [ ] **对话历史**：支持多轮对话上下文
- [ ] **文档预览**：在回答中直接预览文档片段
- [ ] **统计分析**：查询统计、热门问题分析
- [ ] **权限管理**：细粒度的文档访问权限控制

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 贡献方式

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

---

## 🙏 致谢

- [LangChain](https://www.langchain.com/) - RAG 框架
- [Milvus](https://milvus.io/) - 向量数据库
- [MinIO](https://min.io/) - 对象存储
- [OpenAI](https://openai.com/) - Embedding & LLM API

---

## 📮 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 [Issue](https://github.com/your-repo/issues)
- 发送邮件至：your-email@example.com

---

**⭐ 如果这个项目对你有帮助，请给个 Star！**

