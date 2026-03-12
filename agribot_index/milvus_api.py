from enum import Enum
from pymilvus import MilvusClient, DataType
import os
from dotenv import load_dotenv
from logger.logging import setup_logging
from pprint import pprint
from dataclasses import dataclass, asdict

load_dotenv()

logger = setup_logging()

'''Milvus 向量库的读写与检索'''

# 规定：「一条要写入 Milvus 的文档片段」的内存结构。
@dataclass #@dataclass 是 Python 标准库 dataclasses 模块提供的装饰器，会自动为类生成常用方法。
class MilvusDocument:
    doc_name: str # 文档名称
    doc_path_name: str # 文档路径（含名字）
    doc_type: str # 文档类型
    doc_md5: str # 文档MD5
    doc_length: int # 文档字节数
    content: str # 文档分段内容
    content_vector: list # 分段内容向量
    embedding_model: str # embedding模型名称

# 初始化 Milvus 客户端并完成集合/索引/加载。
class MilvusDocumentField(str, Enum):
    ID = "id" # 主键ID
    DOC_NAME = "doc_name" # 文档名称
    DOC_PATH_NAME = "doc_path_name" # 文档路径（含名字）
    DOC_TYPE = "doc_type" # 文档类型
    DOC_MD5 = "doc_md5" # 文档MD5
    DOC_LENGTH = "doc_length" # 文档字节数
    CONTENT = "content" # 文档分段内容
    CONTENT_VECTOR = "content_vector" # 分段内容向量
    EMBEDDING_MODEL = "embedding_model" # embedding模型名称

class MilvusAPI:

    # 初始化 Milvus 客户端并完成集合/索引/加载。
    def __init__(self):
        """初始化Milvus客户端连接"""
        self.database_name = os.getenv("MILVUS_DATABASE")
        self.collection_name = os.getenv("MILVUS_COLLECTION") 
        self.index_name = os.getenv("MILVUS_INDEX_NAME")
        self.vector_dim = int(os.getenv("MILVUS_VECTOR_DIM")) 
        
        if not self.database_name or not self.collection_name or not self.index_name or not self.vector_dim:
            logger.error("Milvus配置错误")
            raise ValueError("Milvus配置错误")

        # 创建客户端连接，指定数据库
        self.client = MilvusClient(
            uri=f"http://{os.getenv('MILVUS_HOST')}:{os.getenv('MILVUS_PORT')}",
            user=os.getenv('MILVUS_USER'),
            password=os.getenv('MILVUS_PASSWORD'),
            db_name=self.database_name
        )
        
        init_result =   self._initialize()
        if not init_result:
            logger.error("Milvus初始化失败")
            raise ValueError("Milvus初始化失败")
                
    # 若集合不存在则创建集合（相当于建表）
    def _create_collection_if_not_exists(self) -> bool:
        """
        创建集合（如果不存在则创建）
        这就相当于在mysql中建表
        表结构为：
            id - 主键ID
            doc_name - 文档名称 （字符串，最大500字符）   "example.pdf"
            doc_path_name -                             "documents/example.pdf"
            doc_type - 文档类型                         "pdf"、"docx"、"txt"、"md"
            doc_md5 - 文档MD5                           用于校验文件完整性 用于去重（判断文件是否被修改）
            doc_length - 文档字节数                      1048576（1MB）、524288（512KB）
            content - 文档分段内容                      存储文档解析后的文本片段内容，每个片段大约128个字符
            content_vector - 分段内容向量                存储文档分段内容向量，用于相似度计算
            embedding_model - embedding模型名称            "text-embedding-v4"
        """
        try:
            # 检查集合是否存在
            if self.client.has_collection(collection_name=self.collection_name):
                logger.info(f"集合 '{self.collection_name}' 已存在")
                return True
            
            # 定义schema
            schema = self.client.create_schema(
                auto_id=True,  # 自动生成ID
                enable_dynamic_field=False
            )
            
            # 添加字段
            # 主键ID字段（自动生成）
            schema.add_field(
                field_name=MilvusDocumentField.ID.value,
                datatype=DataType.INT64,
                is_primary=True,
                auto_id=True
            )
            
            # 文档名称
            schema.add_field(
                field_name=MilvusDocumentField.DOC_NAME.value,
                datatype=DataType.VARCHAR,
                max_length=500
            )
            
            # 文档路径（含名字）
            schema.add_field(
                field_name=MilvusDocumentField.DOC_PATH_NAME.value,
                datatype=DataType.VARCHAR,
                max_length=1000
            )
            
            # 文档类型
            schema.add_field(
                field_name=MilvusDocumentField.DOC_TYPE.value,
                datatype=DataType.VARCHAR,
                max_length=50
            )
            
            # 文档MD5
            schema.add_field(
                field_name=MilvusDocumentField.DOC_MD5.value,
                datatype=DataType.VARCHAR,
                max_length=32
            )
            
            # 文档字节数
            schema.add_field(
                field_name=MilvusDocumentField.DOC_LENGTH.value,
                datatype=DataType.INT64
            )
            
            # 文档内容
            schema.add_field(
                field_name=MilvusDocumentField.CONTENT.value,
                datatype=DataType.VARCHAR,
                max_length=65535  # 最大长度
            )
            
            # 内容向量（当前使用 BAAI/bge-m3，维度默认 1024）
            schema.add_field(
                field_name=MilvusDocumentField.CONTENT_VECTOR.value,
                datatype=DataType.FLOAT_VECTOR,
                dim=self.vector_dim
            )
            
            # embedding模型名称
            schema.add_field(
                field_name=MilvusDocumentField.EMBEDDING_MODEL.value,
                datatype=DataType.VARCHAR,
                max_length=100
            )
            
            # 创建集合
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema
            )
            
            logger.info(f"集合 '{self.collection_name}' 创建成功")
            return True
            
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False
    
    # 若向量索引不存在则创建，用于加速相似度搜索。
    def _create_index_if_not_exists(self) -> bool:
        """创建索引
            1. 给表中的content_vector字段创建索引，索引名字为content_vector
            2. 为甚要给这个字段创建索引呢？
                 因为向量相似性搜索需要索引，如果没有索引的话，搜索的时候就需要遍历所有向量并计算相似度，这样太耗时了
            3. 索引的类型为IVF_FLAT，索引参数为{"nlist": 1024}
            4. 索引参数的含义是：
                nlist: 聚类中心的数量，越大则精度越高，但查询速度越慢
                nprobe: 查询时扫描的聚类中心数量，越大则精度越高，但查询速度越慢
            5. IVF_FLAT索引的工作原理为：
                它是将所有的向量分成若干个聚类中心，然后每个聚类中心存储一个向量，然后查询的时候，先根据查询向量找到最近的聚类中心，然后在这个聚类中心中搜索相似的向量
            
        """
        try:
            # 检查索引是否已存在
            indexes = self.client.list_indexes(collection_name=self.collection_name)
            if self.index_name in indexes:
                logger.info(f"索引 '{self.index_name}' 已存在")
                return True
            
            # 创建向量索引
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name=MilvusDocumentField.CONTENT_VECTOR.value,
                index_type="IVF_FLAT",
                metric_type="COSINE", #相似度计算方式为：余弦相似度
                params={"nlist": 1024}
            )
            
            self.client.create_index(
                collection_name=self.collection_name,
                index_params=index_params
            )
            
            logger.info(f"索引 '{self.index_name}' 创建成功")
            return True
            
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            return False
    
    # 把集合加载到内存，使搜索/查询可用
    def _load_collection(self) -> bool:
        """加载集合到内存"""
        try:
            self.client.load_collection(collection_name=self.collection_name)
            logger.info(f"集合 '{self.collection_name}' 加载成功")
            return True
        except Exception as e:
            logger.error(f"加载集合失败: {e}")
            return False
    
    # 建表 → 建索引 → 加载」的初始化流程。
    def _initialize(self) -> bool:
        """初始化数据库、集合和索引"""
        logger.info("开始初始化Milvus...")
                
        # 创建集合
        if not self._create_collection_if_not_exists():
            return False
        
        # 创建索引
        if not self._create_index_if_not_exists():
            return False
        
        # 加载集合到内存
        if not self._load_collection():
            return False
        
        logger.info("Milvus初始化完成!")
        return True
    
    # 文档路径判断该文档是否已在集合中存在
    def check_document_exists(self, doc_path_name: str) -> bool:
        """
        检查文档是否已存在
        
        Args:
            doc_path_name (str): 文档路径
            
        Returns:
            bool: 文档是否已存在
        """
        try:
            # 先确保集合已加载
            self._load_collection()
            
            # 根据路径查询
            filter_expr = f'doc_path_name == "{doc_path_name}"'
            
            results = self.client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=[MilvusDocumentField.ID.value],
                limit=1
            )
            
            return len(results) > 0
            
        except Exception as e:
            logger.error(f"检查文档是否存在失败: {e}")
            raise e
    
    # 删除该文档路径下的所有记录
    def delete_existing_document(self, doc_path_name: str) -> bool:
        """
        删除已存在的文档记录
        
        Args:
            doc_path_name (str): 文档路径
            
        Returns:
            bool: 删除是否成功
        """
        try:
            # 安全检查：确保doc_path_name不为空，避免删除所有文档
            if not doc_path_name or not doc_path_name.strip():
                logger.error("错误: 文档路径名不能为空，拒绝执行删除操作")
                return False
            
            # 构建删除表达式
            delete_expr = f'doc_path_name == "{doc_path_name}"'
            
            # 执行删除操作
            result = self.client.delete(
                collection_name=self.collection_name,
                filter=delete_expr
            )
            
            logger.info(f"删除已存在的文档记录: {doc_path_name}")
            return True
            
        except Exception as e:
            logger.error(f"删除已存在文档失败: {e}")
            raise e

    # 插入一条文档片段（含标量字段 + 向量）
    def insert_document(self, doc_data: MilvusDocument) -> bool:
        """插入文档数据"""
        try:
            self.client.insert(
                collection_name=self.collection_name,
                data= asdict(doc_data)
            )
            logger.info(f"文档 '{doc_data.doc_name}' 插入成功")
            return True
        except Exception as e:
            logger.error(f"插入文档失败: {e}")
            return False
            
    # 将内存中的写入刷到持久化存储，保证数据落盘。
    def flush_collection(self) -> bool:
        """刷新集合"""
        try:
            self.client.flush(collection_name=self.collection_name)
            logger.info(f"集合 '{self.collection_name}' 刷新成功")
            return True
        except Exception as e:
            logger.error(f"刷新集合失败: {e}")
            return False

    
    # 用一条查询向量做向量相似度搜索，返回最相似的若干条记录。
    def search_similar_documents(self, query_vector, limit=10):
        """搜索相似文档
        
        Args:
            query_vector (list): 查询向量
            limit (int): 返回结果数量限制
            
        Returns:
            list: 搜索结果
        """
        try:
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 64}
            }
            
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                anns_field=MilvusDocumentField.CONTENT_VECTOR.value,
                search_params=search_params,
                limit=limit,
                output_fields=[MilvusDocumentField.DOC_NAME.value, MilvusDocumentField.DOC_PATH_NAME.value, MilvusDocumentField.DOC_TYPE.value, MilvusDocumentField.CONTENT.value, MilvusDocumentField.EMBEDDING_MODEL.value]
            )
            
            return results[0] if results else []
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    # 获取当前集合的描述信息（schema、状态等）
    def get_collection_info(self):
        """获取集合信息"""
        try:
            info = self.client.describe_collection(collection_name=self.collection_name)
            return info
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return None


# 使用示例
if __name__ == "__main__":
    # 创建MilvusAPI实例
    milvus_api = MilvusAPI()
    
    
    # 获取集合信息
    info = milvus_api.get_collection_info()
    if info:
        pprint("集合信息:")
        pprint(info)
