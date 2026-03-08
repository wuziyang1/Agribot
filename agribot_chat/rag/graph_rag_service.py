"""
Graph RAG 服务模块

使用 LangChain + Neo4j 实现知识图谱增强的 RAG 服务。
全流程：
  1. 文档导入：将文本切分 → LLM 抽取实体/关系 → 写入 Neo4j 知识图谱
  2. 图谱查询：用户提问 → LLM 生成 Cypher → 查询图谱 → 结合上下文生成回答
  3. 混合检索：可同时利用向量检索（Milvus）+ 图谱检索（Neo4j）做融合回答
"""

import logging
import queue
import threading
from typing import Dict, Any, List, Optional

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain_community.callbacks.manager import get_openai_callback
from langchain.text_splitter import RecursiveCharacterTextSplitter

from agribot_chat.rag.rag_config import Config

logger = logging.getLogger(__name__)


# =========================================================================
# 辅助函数
# =========================================================================

def _infer_relation_type(entity_type: str) -> str:
    """根据实体类型推断与主体药材之间最合理的关系类型"""
    mapping = {
        "GrowingCondition": "需要条件",
        "CultivationMethod": "种植方式",
        "Disease": "易感染",
        "HumanDisease": "可治疗",
        "Region": "适宜种植于",
        "Season": "种植时间",
        "Soil": "适宜土壤",
        "Fertilizer": "施用",
        "Pesticide": "防治用药为",
        "MedicinalPart": "药用部位为",
        "MedicinalProperty": "药性为",
        "Efficacy": "功效为",
        "ChemicalCompound": "含有成分",
        "HarvestProcess": "采收方式为",
        "ProcessingMethod": "炮制方法为",
        "StorageMethod": "贮藏方式为",
        "Variety": "属于品种",
        "Standard": "质量标准为",
        "Formula": "属于方剂",
        "PlantMorphology": "形态特征为",
        "PlantingPattern": "种植模式为",
        "Concept": "相关",
    }
    return mapping.get(entity_type, "相关")


def _build_doc_context(doc_name: str, main_herbs: list, key_entities: list) -> str:
    """构建文档上下文描述，供 LLM 在抽取时参考"""
    parts = []
    parts.append(f"当前文档名称：{doc_name}")
    if main_herbs:
        parts.append(f"本文档的主体药材：{', '.join(main_herbs[:5])}")
        parts.append(
            "请确保当前文本块中提取的所有实体（尤其是 GrowingCondition、"
            "CultivationMethod、Disease、Region、Season 等）都与上述主体药材建立关系。"
        )
    else:
        parts.append(
            "尚未识别到主体药材。请从当前文本中优先识别出 Herb 类型的主体药材，"
            "并确保其他所有实体都与该主体药材建立关系。"
        )
    if key_entities:
        parts.append(f"前文已识别的关键实体：{', '.join(key_entities[-15:])}")
        parts.append("如果当前文本中的实体与上述已有实体存在关联，请建立相应关系。")
    return "\n".join(parts)


def _filter_orphan_entities(entities: list, relations: list,
                            main_herbs: list) -> tuple:
    """
    过滤孤立实体：只保留在关系中被引用的实体。
    对于未被引用但存在主体药材的情况，自动补充关系。
    """
    referenced = set()
    for rel in relations:
        src = (rel.get("source") or "").strip()
        tgt = (rel.get("target") or "").strip()
        if src:
            referenced.add(src)
        if tgt:
            referenced.add(tgt)

    filtered_entities = []
    new_relations = list(relations)
    for ent in entities:
        ent_name = (ent.get("name") or "").strip()
        if not ent_name:
            continue
        if ent_name in referenced:
            filtered_entities.append(ent)
        elif main_herbs:
            ent_type = (ent.get("type") or "Entity").strip()
            relation_type = _infer_relation_type(ent_type)
            new_relations.append({
                "source": main_herbs[0],
                "source_type": "Herb",
                "target": ent_name,
                "target_type": ent_type,
                "relation": relation_type,
                "properties": {"description": "由系统根据文档上下文自动补充的关系"},
            })
            referenced.add(ent_name)
            referenced.add(main_herbs[0])
            filtered_entities.append(ent)
            logger.debug("自动补充关系: %s -[%s]-> %s", main_herbs[0], relation_type, ent_name)
        else:
            logger.debug("过滤孤立实体: %s", ent_name)

    return filtered_entities, new_relations


# =========================================================================
# 数据模型
# =========================================================================

class GraphEntity(BaseModel):
    """知识图谱实体"""
    name: str
    entity_type: str
    properties: Dict[str, Any] = {}


class GraphRelation(BaseModel):
    """知识图谱关系"""
    source: str
    source_type: str
    target: str
    target_type: str
    relation_type: str
    properties: Dict[str, Any] = {}


class GraphRAGResponse(BaseModel):
    """Graph RAG 响应模型"""
    content: str
    graph_context: str = ""
    entities: List[GraphEntity] = []
    relations: List[GraphRelation] = []
    cypher_query: str = ""
    success: bool = True
    error_message: Optional[str] = None


class GraphImportResult(BaseModel):
    """图谱导入结果"""
    entities_count: int = 0
    relations_count: int = 0
    chunks_processed: int = 0
    success: bool = True
    error_message: Optional[str] = None


# =========================================================================
# 实体/关系抽取提示词
# =========================================================================

EXTRACT_PROMPT = """你是一个中药材种植领域的知识图谱构建专家。请从以下文本中抽取与中药材相关的实体和关系。

## 领域说明
本知识图谱专注于中药材种植领域，涵盖中药材品种、种植技术、病虫害防治、产地与适生区、
采收加工、药用价值、药性归经、化学成分、炮制方法、方剂配伍、植物形态特征等全部内容。

## 文档上下文（极其重要，必须使用）
{doc_context}
**你必须将当前文本中抽取到的每一个实体都与上述主体实体建立关系。**
例如：如果文档主体是"枸杞"，而文本中提到了"温度""湿度""播种"等，
则必须建立"枸杞→温度""枸杞→湿度""枸杞→播种"等关系，绝对不允许这些实体成为孤立节点。

## 实体类型（优先使用以下类型，如遇以下类型无法覆盖但与中药材相关的实体，使用 Concept 类型兜底）
- Herb（中药材）：具体的中药材名称，如黄芪、当归、人参、枸杞、甘草、川芎等
- Variety（品种）：中药材的具体品种或栽培品种，如蒙古黄芪、膜荚黄芪、北柴胡等
- MedicinalPart（药用部位）：根、茎、叶、花、果实、种子、全草、根茎、树皮、块茎等
- Efficacy（功效）：补气、活血、清热、解毒、止咳、化痰、安神、利湿等药理功效
- MedicinalProperty（药性）：四气（寒、热、温、凉、平）、五味（酸、苦、甘、辛、咸）、归经（归肝经、归脾经等）、毒性
- Disease（病虫害）：根腐病、白粉病、锈病、蚜虫、地老虎、红蜘蛛等
- HumanDisease（人体疾病）：感冒、高血压、糖尿病、贫血等中药材可治疗的疾病
- CultivationMethod（种植技术）：播种、育苗、移栽、扦插、分株、组培等栽培技术
- PlantingPattern（种植模式）：轮作、间作、套种、连作、林下种植等
- GrowingCondition（生长条件）：温度、湿度、光照、海拔、pH值、降水量等环境因子
- Region（产地/适生区）：省份、县市、山脉、道地产区，如甘肃岷县、云南文山等
- HarvestProcess（采收加工）：采收时间、采收方法、干燥方式（晒干、阴干、烘干）、初加工
- ProcessingMethod（炮制方法）：酒制、醋制、盐制、蜜炙、炒制、煅制、蒸制等
- ChemicalCompound（化学成分）：黄芪甲苷、多糖、皂苷、生物碱、黄酮、挥发油、有机酸等
- Formula（方剂）：四物汤、六味地黄丸、补中益气汤等经典方剂和现代制剂
- PlantMorphology（植物形态）：株高、叶形、花色、根形等形态鉴别特征
- Fertilizer（肥料）：有机肥、氮肥、磷肥、钾肥、基肥、追肥、叶面肥等
- Pesticide（农药/药剂）：多菌灵、百菌清、吡虫啉等防治用药
- Soil（土壤）：沙壤土、黏土、腐殖土、酸性土壤、碱性土壤等
- Season（时间/季节）：春季、秋季、生长期、休眠期、花期、果期等
- StorageMethod（贮藏方法）：密封保存、阴凉干燥、防潮、防虫蛀等
- Standard（质量标准）：药典标准、等级划分、含量指标、GAP标准等
- Concept（通用概念）：无法归入以上类型但与中药材领域直接相关的重要概念

## 过滤规则（重要）
- **不要提取**：普通人名、作者名、编辑名、出版社、文献编号、页码、ISBN、参考文献等文献元数据
- **不要提取**：与中药材无直接关联的通用概念（如"信息技术""经济发展""市场分析"等）
- **不要提取**：纯数字编号、表格序号等无语义价值的内容
- **保留**：古代医药学家（如张仲景、李时珍、孙思邈）仅在其与具体药材或方剂直接关联时提取，类型设为 Concept
- 每个实体的 name 应尽量简洁准确，避免冗余修饰
- 尽量充分提取，不要遗漏文本中与中药材相关的重要信息

## 关系类型参考（不限于此，可根据文本语义自行命名关系）
- 种植相关：适宜种植于、生长于、需要条件、施用、防治用药为、轮作搭配、间作搭配
- 药用相关：功效为、含有成分、药用部位为、可治疗、配伍、药性为、归经为、属于方剂
- 形态相关：形态特征为、鉴别特征为
- 加工相关：采收方式为、炮制方法为、加工为、贮藏方式为
- 品种相关：属于品种、变种为、别名为、同科属
- 病害相关：易感染、防治方法为、症状为、用药为

## 最重要规则：禁止孤立实体（必须严格遵守）

**每一个实体都必须至少出现在一条关系中（作为 source 或 target）。**
如果文本中没有显式说明某实体的关系，你必须根据上下文推断并建立合理的关系。

推断规则：
1. GrowingCondition（温度、湿度、光照等）→ 必须用"需要条件"关系连接到对应的 Herb
2. CultivationMethod（播种、移栽等）→ 必须用"种植方式"关系连接到对应的 Herb
3. Disease（病虫害）→ 必须用"易感染"关系连接到对应的 Herb
4. Region（产地）→ 必须用"适宜种植于"或"生长于"关系连接到对应的 Herb
5. Season（季节）→ 必须用"种植时间"或"采收时间"关系连接到对应的 Herb
6. 所有其他实体类型 → 必须找到合理的关系连接到文档主体或其他已有实体

**如果你无法为某个实体建立任何关系，则不要提取该实体。**

## 关键要求：数值指标的处理方式（非常重要！）

### 绝对禁止：
- **禁止**将纯数值（如 "25°C"、"60%~65%"、"2~3厘米"、"pH5.5"、"3月"）作为独立实体。
  纯数值不能出现在 entities 的 name 字段中。
- **禁止**创建没有任何关系连接的孤立实体。每个实体至少应出现在一条关系中。

### 正确做法：
数值指标必须通过**关系的 properties** 挂载到有语义的实体上。具体规则：
1. 实体的 name 必须是有语义含义的名词概念（如"温度"、"含水量"、"播种深度"），而不是数值本身。
2. 具体数值放在**关系的 properties.value** 字段中，单位放在 **properties.unit** 字段中。
3. 每条含数值的关系都必须有明确的 source（如某种药材）和 target（如某个条件/方法）。

### 示例（务必严格参照）：

原文："灵芝培养基含水量为60%~65%"
正确抽取：
- entities: [{{"name": "灵芝", "type": "Herb", "properties": {{}}}},
             {{"name": "含水量", "type": "GrowingCondition", "properties": {{"description": "培养基含水量"}}}}]
- relations: [{{"source": "灵芝", "source_type": "Herb",
               "target": "含水量", "target_type": "GrowingCondition",
               "relation": "生长条件", "properties": {{"value": "60%~65%", "description": "培养基含水量"}}}}]
错误示范：❌ 将 "60%~65%" 作为实体的 name

原文："黄芪适宜生长温度为20~25℃，播种深度2~3厘米，土壤pH值6.5~8.0"
正确抽取（每个数值指标一条关系）：
- entities: [{{"name": "黄芪", "type": "Herb", "properties": {{}}}},
             {{"name": "生长温度", "type": "GrowingCondition", "properties": {{}}}},
             {{"name": "播种深度", "type": "CultivationMethod", "properties": {{}}}},
             {{"name": "土壤pH值", "type": "GrowingCondition", "properties": {{}}}}]
- relations: [{{"source": "黄芪", "source_type": "Herb", "target": "生长温度", "target_type": "GrowingCondition",
               "relation": "适宜条件", "properties": {{"value": "20~25", "unit": "℃"}}}},
              {{"source": "黄芪", "source_type": "Herb", "target": "播种深度", "target_type": "CultivationMethod",
               "relation": "种植技术", "properties": {{"value": "2~3", "unit": "厘米"}}}},
              {{"source": "黄芪", "source_type": "Herb", "target": "土壤pH值", "target_type": "GrowingCondition",
               "relation": "适宜条件", "properties": {{"value": "6.5~8.0"}}}}]
错误示范：❌ 将 "20~25℃"、"2~3厘米"、"6.5~8.0" 作为实体的 name

原文："灵芝子实体生长阶段温度控制在25~28℃，空气湿度85%~95%，CO2浓度低于0.1%"
正确抽取（三条关系，数值全部在 properties 中）：
- relations: [{{"source": "灵芝", ..., "target": "子实体生长温度", "target_type": "GrowingCondition",
               "relation": "生长条件", "properties": {{"value": "25~28", "unit": "℃", "description": "子实体生长阶段"}}}},
              {{"source": "灵芝", ..., "target": "空气湿度", "target_type": "GrowingCondition",
               "relation": "生长条件", "properties": {{"value": "85%~95%", "description": "子实体生长阶段"}}}},
              {{"source": "灵芝", ..., "target": "CO2浓度", "target_type": "GrowingCondition",
               "relation": "生长条件", "properties": {{"value": "低于0.1%", "description": "子实体生长阶段"}}}}]

## 输出格式
严格遵循以下 JSON 格式，不要输出多余文字：
```json
{{
  "entities": [
    {{"name": "实体名称", "type": "实体类型", "properties": {{"description": "简要描述", "value": "具体数值（如有）", "unit": "单位（如有）"}}}}
  ],
  "relations": [
    {{"source": "源实体名称", "source_type": "源实体类型",
      "target": "目标实体名称", "target_type": "目标实体类型",
      "relation": "关系类型", "properties": {{"value": "具体数值（如有）", "unit": "单位（如有）", "description": "补充说明（如有）"}}}}
  ]
}}
```

文本内容：
{text}

请充分抽取上述文本中与中药材相关的所有实体和关系。
重要提醒：具体数值（温度、含水量、pH值、浓度、用量等）绝不能作为实体名称，只能放在关系的 properties.value 中。
每个实体必须至少出现在一条关系中，不允许孤立节点。
以 JSON 格式输出："""


# =========================================================================
# Cypher 生成提示词
# =========================================================================

CYPHER_GENERATION_PROMPT = """你是一个 Neo4j Cypher 查询专家，专注于中药材种植领域的知识图谱查询。
请根据用户问题和图谱 Schema 生成 Cypher 查询语句。

图谱 Schema：
{schema}

图谱中的节点类型主要包括：Herb（中药材）、Variety（品种）、MedicinalPart（药用部位）、
Efficacy（功效）、MedicinalProperty（药性）、Disease（病虫害）、HumanDisease（人体疾病）、
CultivationMethod（种植技术）、PlantingPattern（种植模式）、GrowingCondition（生长条件）、
Region（产地/适生区）、HarvestProcess（采收加工）、ProcessingMethod（炮制方法）、
ChemicalCompound（化学成分）、Formula（方剂）、PlantMorphology（植物形态）、
Fertilizer（肥料）、Pesticide（农药/药剂）、Soil（土壤）、Season（时间/季节）、
StorageMethod（贮藏方法）、Standard（质量标准）、Concept（通用概念）。

注意事项：
1. 只生成 READ 查询（MATCH），不要生成任何写入/删除操作
2. 使用 CONTAINS 进行模糊匹配（中文实体名称可能不完全一致）
3. 限制返回结果数量（LIMIT 20）
4. 返回节点的名称、类型、关系信息，以及节点和关系的全部属性（使用 properties() 函数），
   确保具体数值（如含水量、温度、pH值、浓度、剂量等）不会丢失
5. 只输出 Cypher 语句，不要输出其他内容
6. 当用户询问某种中药材时，优先查询以该药材为中心的所有关联节点和关系

用户问题：{question}

Cypher 查询语句："""


# =========================================================================
# 图谱上下文回答提示词
# =========================================================================

GRAPH_QA_PROMPT = """你是一个基于知识图谱的问答助手。请根据知识图谱查询结果回答用户问题。

知识图谱查询结果：
{graph_context}

以下是本对话的近期历史（请结合历史理解并回答当前问题）：
{chat_history}

用户问题：{question}

回答要求：
1. 基于知识图谱中的实体和关系进行回答
2. 如果图谱信息不足以回答，请如实说明并结合通用知识补充
3. 回答使用自然流畅的中文，语言简洁、逻辑清晰
4. 可以合理使用 Markdown 语法提升可读性
5. 若有近期对话历史，请结合上下文保持连贯

请给出回答："""


# =========================================================================
# GraphRAGService 主类
# =========================================================================

class GraphRAGService:
    """
    Graph RAG 服务类

    功能：
    1. 文档导入知识图谱：文本 -> LLM 抽取实体/关系 -> Neo4j
    2. 图谱问答：用户提问 -> Cypher 查询 -> LLM 生成回答
    3. 图谱管理：查看统计、清空图谱等
    """

    CHAT_HISTORY_MAX_MESSAGES = 10

    def __init__(self):
        self.graph: Optional[Neo4jGraph] = None
        self.llm: Optional[ChatOpenAI] = None
        self._initialize()

    def _initialize(self):
        """初始化Neo4j连接和LLM"""
        try:
            # 初始化 Neo4j 图数据库连接
            self.graph = Neo4jGraph(
                url=Config.NEO4J_URI,
                username=Config.NEO4J_USERNAME,
                password=Config.NEO4J_PASSWORD,
                database=Config.NEO4J_DATABASE,
            )
            # 刷新 schema
            self.graph.refresh_schema()
            logger.info("Neo4j 连接成功: %s", Config.NEO4J_URI)

            # 初始化 LLM
            self.llm = ChatOpenAI(
                model=Config.LLM_MODEL_NAME,
                openai_api_key=Config.LLM_API_KEY,
                openai_api_base=Config.LLM_BASE_URL,
                temperature=0.0,
                max_tokens=2000,
            )
            logger.info("Graph RAG 服务初始化完成")

        except Exception as e:
            logger.error("Graph RAG 服务初始化失败: %s", e)
            raise

    def _create_llm(self, *, streaming=False, callbacks=None):
        """创建 LLM 客户端（按需开启 streaming）"""
        kwargs = {
            "model": Config.LLM_MODEL_NAME,
            "openai_api_key": Config.LLM_API_KEY,
            "openai_api_base": Config.LLM_BASE_URL,
            "temperature": 0.0,
            "max_tokens": 2000,
        }
        if streaming:
            kwargs["streaming"] = True
        if callbacks:
            kwargs["callbacks"] = callbacks
        return ChatOpenAI(**kwargs)

    def _format_chat_history(self, chat_history=None):
        if not chat_history:
            return "（无近期历史）"
        lines = []
        for msg in chat_history[-self.CHAT_HISTORY_MAX_MESSAGES:]:
            role = (msg.get("role") or "").strip().lower()
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                lines.append("用户：" + content)
            else:
                lines.append("助手：" + content)
        return "\n".join(lines) if lines else "（无近期历史）"

    # =================================================================
    # 文档导入知识图谱
    # =================================================================

    def import_text(self, text, doc_name="", chunk_size=1000, chunk_overlap=200):
        """将文本导入知识图谱

        流程：文本切分 -> LLM 抽取实体/关系 -> 写入 Neo4j
        """
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", "。", "；", ".", ";", " "]
            )
            chunks = splitter.split_text(text)
            logger.info("文本切分完成，共 %d 个块", len(chunks))

            total_entities = 0
            total_relations = 0

            # 跨块上下文：追踪文档主体实体
            main_herbs = []
            all_key_entities = []

            for i, chunk in enumerate(chunks):
                try:
                    doc_context = _build_doc_context(doc_name, main_herbs, all_key_entities)
                    extracted = self._extract_entities_relations(chunk, doc_context=doc_context)
                    if not extracted:
                        continue

                    entities = extracted.get("entities", [])
                    relations = extracted.get("relations", [])

                    # 收集主体实体
                    for ent in entities:
                        ent_name = (ent.get("name") or "").strip()
                        ent_type = (ent.get("type") or "").strip()
                        if ent_type == "Herb" and ent_name and ent_name not in main_herbs:
                            main_herbs.append(ent_name)
                        if ent_name and ent_name not in all_key_entities:
                            all_key_entities.append(ent_name)

                    # 过滤孤立实体
                    entities, relations = _filter_orphan_entities(entities, relations, main_herbs)

                    chunk_id = f"{doc_name}#chunk_{i}"
                    self._write_to_neo4j(entities, relations, doc_name, chunk_id)
                    total_entities += len(entities)
                    total_relations += len(relations)

                    logger.info(
                        "块 %d/%d: 抽取 %d 实体, %d 关系",
                        i + 1, len(chunks), len(entities), len(relations)
                    )
                except Exception as e:
                    logger.warning("块 %d 处理失败: %s", i + 1, e)
                    continue

            # 导入完成后，清理该文档产生的孤立节点
            try:
                orphan_result = self.graph.query(
                    "MATCH (n) WHERE n.doc_source = $doc "
                    "AND NOT EXISTS { MATCH (n)-[]-() } "
                    "WITH n, n.name AS name "
                    "DELETE n "
                    "RETURN count(name) AS deleted_count",
                    params={"doc": doc_name},
                )
                orphan_count = orphan_result[0]["deleted_count"] if orphan_result else 0
                if orphan_count > 0:
                    logger.info("图谱清理 - 已删除 %d 个孤立节点 [%s]", orphan_count, doc_name)
            except Exception as e:
                logger.warning("清理孤立节点失败: %s", e)

            self.graph.refresh_schema()

            logger.info(
                "图谱导入完成: %d 实体, %d 关系（来自 %d 个文本块）",
                total_entities, total_relations, len(chunks)
            )
            return GraphImportResult(
                entities_count=total_entities,
                relations_count=total_relations,
                chunks_processed=len(chunks),
                success=True,
            )

        except Exception as e:
            logger.error("图谱导入失败: %s", e)
            return GraphImportResult(success=False, error_message=str(e))

    def _extract_entities_relations(self, text, doc_context=""):
        """使用 LLM 从文本中抽取实体和关系"""
        import json as _json

        if not doc_context:
            doc_context = "无额外上下文信息。请从文本中自行识别主体药材并确保所有实体与之关联。"
        prompt = EXTRACT_PROMPT.format(text=text, doc_context=doc_context)
        try:
            result = self.llm.invoke(prompt)
            content = result.content.strip()

            # 尝试从 markdown 代码块中提取 JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            return _json.loads(content)

        except Exception as e:
            logger.warning("实体关系抽取失败: %s", e)
            return None

    def _write_to_neo4j(self, entities, relations, doc_name="", chunk_id=""):
        """将抽取的实体和关系写入 Neo4j，同时绑定 chunk_id"""
        for ent in entities:
            name = ent.get("name", "").strip()
            ent_type = ent.get("type", "Entity").strip()
            props = ent.get("properties", {})
            if not name:
                continue

            ent_type = "".join(c for c in ent_type if c.isalnum() or c == "_") or "Entity"

            # 构建参数：基础字段 + 抽取到的所有 properties
            params = {
                "name": name,
                "doc_source": doc_name,
                "chunk_id": chunk_id,
            }
            set_parts = ["n.doc_source = $doc_source", "n.chunk_id = $chunk_id"]
            for pk, pv in props.items():
                safe_key = "".join(c for c in pk if c.isalnum() or c == "_") or "prop"
                param_key = f"prop_{safe_key}"
                params[param_key] = str(pv) if pv is not None else ""
                set_parts.append(f"n.{safe_key} = ${param_key}")

            cypher = (
                f"MERGE (n:`{ent_type}` {{name: $name}}) "
                f"SET {', '.join(set_parts)}"
            )
            try:
                self.graph.query(cypher, params=params)
            except Exception as e:
                logger.warning("写入实体失败 [%s]: %s", name, e)

        for rel in relations:
            src = rel.get("source", "").strip()
            tgt = rel.get("target", "").strip()
            src_type = rel.get("source_type", "Entity").strip()
            tgt_type = rel.get("target_type", "Entity").strip()
            rel_type = rel.get("relation", "RELATED_TO").strip()
            if not src or not tgt:
                continue

            src_type = "".join(c for c in src_type if c.isalnum() or c == "_") or "Entity"
            tgt_type = "".join(c for c in tgt_type if c.isalnum() or c == "_") or "Entity"
            rel_type = "".join(c for c in rel_type if c.isalnum() or c == "_") or "RELATED_TO"

            rel_props = rel.get("properties", {}) or {}
            params = {
                "src": src, "tgt": tgt,
                "doc_source": doc_name, "chunk_id": chunk_id,
            }
            set_parts = ["r.doc_source = $doc_source", "r.chunk_id = $chunk_id"]
            for pk, pv in rel_props.items():
                safe_key = "".join(c for c in pk if c.isalnum() or c == "_") or "prop"
                param_key = f"rprop_{safe_key}"
                params[param_key] = str(pv) if pv is not None else ""
                set_parts.append(f"r.{safe_key} = ${param_key}")

            cypher = (
                f"MERGE (a:`{src_type}` {{name: $src}}) "
                f"MERGE (b:`{tgt_type}` {{name: $tgt}}) "
                f"MERGE (a)-[r:`{rel_type}`]->(b) "
                f"SET {', '.join(set_parts)}"
            )
            try:
                self.graph.query(cypher, params=params)
            except Exception as e:
                logger.warning("写入关系失败 [%s->%s]: %s", src, tgt, e)

    # =================================================================
    # 图谱问答
    # =================================================================

    def query(self, question, chat_history=None):
        """知识图谱问答

        流程：用户提问 -> LLM 生成 Cypher -> 查询图谱 -> LLM 生成回答
        """
        try:
            if not question or not question.strip():
                return GraphRAGResponse(
                    content="请输入有效的查询内容",
                    success=False,
                    error_message="查询内容为空"
                )

            logger.info("Graph RAG 查询: %s", question)

            self.graph.refresh_schema()
            schema = self.graph.schema

            cypher_query = self._generate_cypher(question, schema)
            logger.info("生成 Cypher: %s", cypher_query)

            graph_results = []
            if cypher_query:
                try:
                    graph_results = self.graph.query(cypher_query)
                    logger.info("图谱返回 %d 条结果", len(graph_results))
                except Exception as e:
                    logger.warning("Cypher 执行失败: %s — 回退到模糊搜索", e)
                    graph_results = self._fallback_search(question)

            if not graph_results:
                graph_results = self._fallback_search(question)

            graph_context = self._format_graph_results(graph_results)

            chat_history_str = self._format_chat_history(chat_history)
            qa_prompt = GRAPH_QA_PROMPT.format(
                graph_context=graph_context,
                chat_history=chat_history_str,
                question=question,
            )

            with get_openai_callback() as cb:
                answer = self.llm.invoke(qa_prompt).content

            logger.info("Graph RAG 查询完成")

            entities, relations = self._extract_from_results(graph_results)

            return GraphRAGResponse(
                content=answer,
                graph_context=graph_context,
                entities=entities,
                relations=relations,
                cypher_query=cypher_query or "",
                success=True,
            )

        except Exception as e:
            logger.error("Graph RAG 查询失败: %s", e)
            return GraphRAGResponse(
                content="",
                success=False,
                error_message=f"图谱查询失败：{e}"
            )

    def stream_query(self, question, chat_history=None):
        """流式图谱问答，逐 token yield 事件字典（与 RAGService.stream_query 格式一致）"""
        try:
            from langchain_core.callbacks.base import BaseCallbackHandler
        except Exception:
            from langchain.callbacks.base import BaseCallbackHandler

        yield {"type": "start"}

        if not question or not question.strip():
            yield {
                "type": "error",
                "data": {
                    "success": False,
                    "content": "",
                    "error_message": "查询内容为空",
                    "source_documents": [],
                    "token_usage": None,
                },
            }
            return

        # 1-3: 检索图谱（同步部分）
        try:
            self.graph.refresh_schema()
            schema = self.graph.schema
            cypher_query = self._generate_cypher(question, schema)
            logger.info("Stream Cypher: %s", cypher_query)

            graph_results = []
            if cypher_query:
                try:
                    graph_results = self.graph.query(cypher_query)
                except Exception as e:
                    logger.warning("Cypher 执行失败: %s", e)
                    graph_results = self._fallback_search(question)
            if not graph_results:
                graph_results = self._fallback_search(question)

            graph_context = self._format_graph_results(graph_results)
            entities, relations = self._extract_from_results(graph_results)
        except Exception as e:
            yield {
                "type": "error",
                "data": {
                    "success": False,
                    "content": "",
                    "error_message": f"图谱检索失败：{e}",
                    "source_documents": [],
                    "token_usage": None,
                },
            }
            return

        # 4: 流式 LLM 生成
        chat_history_str = self._format_chat_history(chat_history)
        qa_prompt = GRAPH_QA_PROMPT.format(
            graph_context=graph_context,
            chat_history=chat_history_str,
            question=question,
        )

        _STOP = object()
        token_q = queue.Queue()
        state = {"answer": "", "token_usage": None, "error": None}

        class _TokenQueueCB(BaseCallbackHandler):
            def on_llm_new_token(self, token, **kwargs):
                if token:
                    token_q.put(token)

        def _worker():
            try:
                cb_handler = _TokenQueueCB()
                llm_s = self._create_llm(streaming=True, callbacks=[cb_handler])
                with get_openai_callback() as cb:
                    result = llm_s.invoke(qa_prompt)
                state["answer"] = getattr(result, "content", "") or ""
                state["token_usage"] = {
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_tokens": cb.total_tokens,
                }
            except Exception as e:
                state["error"] = e
            finally:
                token_q.put(_STOP)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        while True:
            item = token_q.get()
            if item is _STOP:
                break
            yield {"type": "chunk", "data": {"content": str(item)}}

        if state["error"] is not None:
            yield {
                "type": "error",
                "data": {
                    "success": False,
                    "content": "",
                    "error_message": f"图谱问答生成失败：{state['error']}",
                    "source_documents": [],
                    "token_usage": None,
                },
            }
            return

        # 将图谱实体/关系以 source_documents 格式返回
        source_docs = []
        for ent in entities:
            source_docs.append({
                "doc_name": f"[{ent.entity_type}] {ent.name}",
                "doc_path_name": "",
                "doc_type": "graph_entity",
                "content_preview": ent.properties.get("description", ent.name),
                "similarity_score": None,
            })

        yield {
            "type": "end",
            "data": {
                "success": True,
                "content": state.get("answer") or "",
                "error_message": None,
                "source_documents": source_docs[:10],
                "token_usage": state.get("token_usage"),
                "graph_info": {
                    "cypher_query": cypher_query or "",
                    "entities_count": len(entities),
                    "relations_count": len(relations),
                },
            },
        }

    # =================================================================
    # 图谱管理
    # =================================================================

    def get_stats(self):
        """获取知识图谱统计信息"""
        try:
            node_count = self.graph.query("MATCH (n) RETURN count(n) AS cnt")
            rel_count = self.graph.query("MATCH ()-[r]->() RETURN count(r) AS cnt")
            labels = self.graph.query("CALL db.labels() YIELD label RETURN collect(label) AS labels")
            rel_types = self.graph.query(
                "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) AS types"
            )

            return {
                "node_count": node_count[0]["cnt"] if node_count else 0,
                "relation_count": rel_count[0]["cnt"] if rel_count else 0,
                "labels": labels[0]["labels"] if labels else [],
                "relationship_types": rel_types[0]["types"] if rel_types else [],
                "schema": self.graph.schema,
            }
        except Exception as e:
            logger.error("获取图谱统计失败: %s", e)
            return {"error": str(e)}

    def clear_graph(self):
        """清空知识图谱（谨慎操作）"""
        try:
            self.graph.query("MATCH (n) DETACH DELETE n")
            self.graph.refresh_schema()
            logger.info("知识图谱已清空")
            return True
        except Exception as e:
            logger.error("清空图谱失败: %s", e)
            return False

    def health_check(self):
        """健康检查"""
        try:
            result = self.graph.query("RETURN 1 AS ok")
            ok = bool(result and result[0].get("ok") == 1)
            return {
                "service": "GraphRAGService",
                "status": "healthy" if ok else "error",
                "neo4j_uri": Config.NEO4J_URI,
            }
        except Exception as e:
            return {
                "service": "GraphRAGService",
                "status": "error",
                "error": str(e),
            }

    # =================================================================
    # 私有辅助方法
    # =================================================================

    def _generate_cypher(self, question, schema):
        """LLM 生成 Cypher 查询"""
        prompt = CYPHER_GENERATION_PROMPT.format(schema=schema, question=question)
        try:
            result = self.llm.invoke(prompt)
            cypher = result.content.strip()
            # 去掉可能的 markdown 代码块标记
            if cypher.startswith("```"):
                lines = cypher.split("\n")
                cypher = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
                cypher = cypher.strip()

            # 安全检查：只允许读操作
            upper = cypher.upper()
            forbidden = ["DELETE", "REMOVE", "SET ", "CREATE", "MERGE", "DROP"]
            for kw in forbidden:
                if kw in upper:
                    logger.warning("Cypher 包含危险关键字 '%s'，已拒绝执行", kw)
                    return None
            return cypher
        except Exception as e:
            logger.warning("Cypher 生成失败: %s", e)
            return None

    def _fallback_search(self, question):
        """回退搜索：用关键词在图谱中模糊匹配（包含节点和关系属性）"""
        try:
            results = self.graph.query(
                "MATCH (n) WHERE n.name CONTAINS $keyword "
                "OPTIONAL MATCH (n)-[r]-(m) "
                "RETURN n.name AS source_name, labels(n) AS source_labels, "
                "properties(n) AS source_props, "
                "type(r) AS relation, properties(r) AS relation_props, "
                "m.name AS target_name, labels(m) AS target_labels, "
                "properties(m) AS target_props "
                "LIMIT 20",
                params={"keyword": question[:20]},
            )
            return results
        except Exception as e:
            logger.warning("回退搜索失败: %s", e)
            return []

    def _format_graph_results(self, results):
        """将图谱查询结果格式化为可读文本（包含属性中的具体数值）"""
        if not results:
            return "（知识图谱中未找到相关信息）"

        lines = []
        for i, record in enumerate(results[:20]):
            parts = []
            for k, v in record.items():
                if v is None:
                    continue
                # 如果值是 Neo4j 节点/关系对象，展开其属性
                if hasattr(v, "items"):
                    detail = ", ".join(f"{pk}={pv}" for pk, pv in v.items() if pv)
                    parts.append(f"{k}: {{{detail}}}")
                elif isinstance(v, list) and v and hasattr(v[0], "items"):
                    for item in v:
                        detail = ", ".join(f"{pk}={pv}" for pk, pv in item.items() if pv)
                        parts.append(f"{k}: {{{detail}}}")
                else:
                    parts.append(f"{k}: {v}")
            if parts:
                lines.append(f"  {i + 1}. " + " | ".join(parts))

        return "知识图谱查询结果：\n" + "\n".join(lines) if lines else "（无结果）"

    def _extract_from_results(self, results):
        """从查询结果中提取实体和关系"""
        entities_set = set()
        entities = []
        relations = []

        for record in results:
            for key in ("source_name", "name", "target_name"):
                name = record.get(key)
                if name and name not in entities_set:
                    entities_set.add(name)
                    label_key = key.replace("name", "labels")
                    labels_val = record.get(label_key, [])
                    etype = labels_val[0] if isinstance(labels_val, list) and labels_val else "Entity"
                    entities.append(GraphEntity(name=name, entity_type=etype))

            src = record.get("source_name")
            tgt = record.get("target_name")
            rel = record.get("relation") or record.get("type(r)")
            if src and tgt and rel:
                relations.append(GraphRelation(
                    source=src, source_type="Entity",
                    target=tgt, target_type="Entity",
                    relation_type=rel,
                ))

        return entities, relations


# =========================================================================
# 单例管理
# =========================================================================

_graph_rag_instance = None


def get_graph_rag_service():
    """获取 GraphRAGService 单例（仅在配置了 Neo4j 时创建）"""
    global _graph_rag_instance
    if _graph_rag_instance is not None:
        return _graph_rag_instance

    if not Config.NEO4J_URI:
        logger.info("未配置 NEO4J_URI，Graph RAG 服务不可用")
        return None

    try:
        _graph_rag_instance = GraphRAGService()
        logger.info("Graph RAG 服务实例创建成功")
        return _graph_rag_instance
    except Exception as e:
        logger.error("Graph RAG 服务实例创建失败: %s", e)
        return None
