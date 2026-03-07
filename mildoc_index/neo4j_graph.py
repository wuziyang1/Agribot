"""
知识图谱索引模块

在文档解析入库时，同步将文本导入 Neo4j 知识图谱。
流程：文本块 → LLM 抽取实体/关系 → 写入 Neo4j

如果 Neo4j 相关环境变量未配置，则自动跳过（不影响向量索引功能）。
"""

import json
import os
import logging
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from logger.logging import setup_logging

load_dotenv()
logger = setup_logging()


# =========================================================================
# 实体/关系抽取提示词
# =========================================================================

EXTRACT_PROMPT = """你是一个专业的知识图谱构建专家。请从以下文本中抽取实体和关系。

要求：
1. 实体类型包括但不限于：Person（人物）、Organization（组织）、Location（地点）、
   Event（事件）、Product（产品）、Technology（技术）、Concept（概念）、Document（文档）、
   Date（日期）、Amount（金额）
2. 关系应描述实体间的语义关联
3. 输出严格遵循以下 JSON 格式，不要输出多余文字

输出格式：
```json
{{
  "entities": [
    {{"name": "实体名称", "type": "实体类型", "properties": {{"description": "简要描述"}}}}
  ],
  "relations": [
    {{"source": "源实体名称", "source_type": "源实体类型",
      "target": "目标实体名称", "target_type": "目标实体类型",
      "relation": "关系类型", "properties": {{}}}}
  ]
}}
```

文本内容：
{text}

请抽取上述文本中的实体和关系，以 JSON 格式输出："""


class GraphIndexer:
    """
    知识图谱索引器

    在 mildoc_index 流程中被调用，负责：
    1. 接收已解析的文本块列表
    2. 调用 LLM 抽取实体/关系
    3. 写入 Neo4j 知识图谱
    """

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 neo4j_database: str = "neo4j"):
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self._database = neo4j_database

        # 用于 LLM 抽取实体/关系（复用 mildoc_index 的 OpenAI 配置，或使用专用 LLM 配置）
        llm_api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        llm_base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL", "")
        self._llm_model = os.getenv("LLM_MODEL_NAME", "qwen-plus")
        self._llm_client = OpenAI(api_key=llm_api_key, base_url=llm_base_url)

        # 验证连接
        self._verify_connection()
        logger.info("Neo4j 知识图谱索引器初始化成功: %s", neo4j_uri)

    def _verify_connection(self):
        """验证 Neo4j 连接是否可用"""
        with self._driver.session(database=self._database) as session:
            session.run("RETURN 1")

    def close(self):
        """关闭驱动"""
        if self._driver:
            self._driver.close()

    # ------------------------------------------------------------------
    # 核心：导入文档到知识图谱
    # ------------------------------------------------------------------

    def import_document(self, doc_name: str, doc_path_name: str,
                        text_chunks: List[str]) -> Dict:
        """
        将一个文档的所有文本块导入知识图谱

        Args:
            doc_name: 文档名称
            doc_path_name: 文档路径（MinIO 中的 object_name）
            text_chunks: 文本块列表（已由 SimpleObjectParser 切分好的）

        Returns:
            dict: {"entities_count": int, "relations_count": int, "success": bool}
        """
        total_entities = 0
        total_relations = 0

        for i, chunk in enumerate(text_chunks):
            chunk_id = f"{doc_path_name}#chunk_{i}"
            try:
                extracted = self._extract_entities_relations(chunk)
                if not extracted:
                    continue

                entities = extracted.get("entities", [])
                relations = extracted.get("relations", [])

                self._write_to_neo4j(entities, relations, doc_name, doc_path_name, chunk_id)
                total_entities += len(entities)
                total_relations += len(relations)

                logger.info(
                    "  图谱导入 - 块 %d/%d: %d 实体, %d 关系",
                    i + 1, len(text_chunks), len(entities), len(relations),
                )
            except Exception as e:
                logger.warning("  图谱导入 - 块 %d 处理失败: %s", i + 1, e)
                continue

        logger.info(
            "  图谱导入完成 [%s]: %d 实体, %d 关系",
            doc_name, total_entities, total_relations,
        )
        return {
            "entities_count": total_entities,
            "relations_count": total_relations,
            "chunks_processed": len(text_chunks),
            "success": True,
        }

    def delete_document(self, doc_path_name: str) -> bool:
        """
        从知识图谱中删除某个文档相关的所有实体和关系

        删除策略：删除所有 doc_source 等于该文档路径的节点和关系。
        如果某个节点同时被其他文档引用（关联了多个 doc_source），则只移除属性标记，不删除节点。

        Args:
            doc_path_name: 文档路径

        Returns:
            bool: 是否成功
        """
        try:
            with self._driver.session(database=self._database) as session:
                # 先删除只属于该文档的关系
                session.run(
                    "MATCH ()-[r]->() WHERE r.doc_source = $doc "
                    "DELETE r",
                    doc=doc_path_name,
                )
                # 再删除只属于该文档且无其他关系的孤立节点
                session.run(
                    "MATCH (n) WHERE n.doc_source = $doc "
                    "AND NOT EXISTS { MATCH (n)-[]-() } "
                    "DELETE n",
                    doc=doc_path_name,
                )
            logger.info("图谱中已删除文档 [%s] 的相关数据", doc_path_name)
            return True
        except Exception as e:
            logger.error("图谱删除文档失败 [%s]: %s", doc_path_name, e)
            return False

    # ------------------------------------------------------------------
    # LLM 抽取实体/关系
    # ------------------------------------------------------------------

    def _extract_entities_relations(self, text: str) -> Optional[Dict]:
        """使用 LLM 从文本中抽取实体和关系"""
        prompt = EXTRACT_PROMPT.format(text=text)
        try:
            response = self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2000,
            )
            content = response.choices[0].message.content.strip()

            # 尝试从 markdown 代码块中提取 JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            return json.loads(content)
        except Exception as e:
            logger.warning("实体关系抽取失败: %s", e)
            return None

    # ------------------------------------------------------------------
    # 写入 Neo4j
    # ------------------------------------------------------------------

    def _write_to_neo4j(self, entities: List[Dict], relations: List[Dict],
                        doc_name: str, doc_path_name: str, chunk_id: str = ""):
        """将抽取的实体和关系写入 Neo4j，同时绑定 chunk_id 以打通向量索引"""
        with self._driver.session(database=self._database) as session:
            # 写入实体
            for ent in entities:
                name = (ent.get("name") or "").strip()
                ent_type = (ent.get("type") or "Entity").strip()
                props = ent.get("properties", {})
                if not name:
                    continue
                # 清理类型名称，确保是合法的 Neo4j 标签
                ent_type = "".join(c for c in ent_type if c.isalnum() or c == "_") or "Entity"
                description = props.get("description", "")

                try:
                    session.run(
                        f"MERGE (n:`{ent_type}` {{name: $name}}) "
                        f"SET n.doc_source = $doc_source, "
                        f"n.doc_name = $doc_name, "
                        f"n.description = $description, "
                        f"n.chunk_id = $chunk_id",
                        name=name,
                        doc_source=doc_path_name,
                        doc_name=doc_name,
                        description=description,
                        chunk_id=chunk_id,
                    )
                except Exception as e:
                    logger.warning("写入实体失败 [%s]: %s", name, e)

            # 写入关系
            for rel in relations:
                src = (rel.get("source") or "").strip()
                tgt = (rel.get("target") or "").strip()
                src_type = (rel.get("source_type") or "Entity").strip()
                tgt_type = (rel.get("target_type") or "Entity").strip()
                rel_type = (rel.get("relation") or "RELATED_TO").strip()
                if not src or not tgt:
                    continue

                src_type = "".join(c for c in src_type if c.isalnum() or c == "_") or "Entity"
                tgt_type = "".join(c for c in tgt_type if c.isalnum() or c == "_") or "Entity"
                rel_type = "".join(c for c in rel_type if c.isalnum() or c == "_") or "RELATED_TO"

                try:
                    session.run(
                        f"MERGE (a:`{src_type}` {{name: $src}}) "
                        f"MERGE (b:`{tgt_type}` {{name: $tgt}}) "
                        f"MERGE (a)-[r:`{rel_type}`]->(b) "
                        f"SET r.doc_source = $doc_source, "
                        f"r.chunk_id = $chunk_id",
                        src=src,
                        tgt=tgt,
                        doc_source=doc_path_name,
                        chunk_id=chunk_id,
                    )
                except Exception as e:
                    logger.warning("写入关系失败 [%s->%s]: %s", src, tgt, e)


# =========================================================================
# 工厂函数（供 main.py 调用）
# =========================================================================

def create_graph_indexer() -> Optional[GraphIndexer]:
    """
    根据环境变量创建 GraphIndexer 实例。
    如果 NEO4J_URI 未配置，则返回 None（知识图谱功能跳过）。
    """
    neo4j_uri = os.getenv("NEO4J_URI", "")
    if not neo4j_uri:
        logger.info("未配置 NEO4J_URI，知识图谱索引功能已跳过")
        return None

    neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "")
    neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")

    try:
        indexer = GraphIndexer(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            neo4j_database=neo4j_database,
        )
        return indexer
    except Exception as e:
        logger.error("知识图谱索引器创建失败: %s（将跳过图谱索引）", e)
        return None
