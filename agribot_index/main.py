import json
import threading
import time
from datetime import datetime
from typing import Dict, Any, List
import os
import argparse
from dotenv import load_dotenv

from minio import Minio
from parser.simple_object_parser import SimpleObjectParser
from embedding import EmbeddingTool
from milvus_api import MilvusAPI, MilvusDocument
from neo4j_graph import create_graph_indexer
from logger.logging import setup_logging

load_dotenv()

logger = setup_logging()


class MinioEventListener:
    """Minio事件监听器"""
    
    def __init__(self, bucket_name: str = None):
        """
        初始化监听器
        
        Args:
            bucket_name (str): 要监听的桶名称，默认从环境变量获取
        """
        self.bucket_name = bucket_name or os.getenv("MINIO_BUCKET", "public-docs")
        #这个名为test的桶是在minio中提前创建好的
        
        # 初始化各个组件
        self.minio_client = Minio(
            endpoint=os.getenv("ENDPOINT"),
            access_key=os.getenv("ACCESS_KEY"),
            secret_key=os.getenv("SECRET_KEY"),
            secure=False
        )
        
        # 初始化解析器
        logger.info("初始化解析器...")
        self.parser: SimpleObjectParser = SimpleObjectParser()
        #冒号的作用：变量类型注解。parser变量属于SimpleObjectParser类型 默认值为 SimpleObjectParser()
        # SimpleObjectParser()这是实例化类，也就是说self.parser是 SimpleObjectParser()类的对象
        
        # 初始化Milvus
        logger.info("初始化Milvus...")
        self.milvus_api: MilvusAPI = MilvusAPI()
        
        # 测试embedding工具
        logger.info("测试embedding工具...")
        self.embedding_tool: EmbeddingTool = EmbeddingTool()


        # 初始化知识图谱索引器（可选，缺少 Neo4j 配置时自动跳过）
        logger.info("初始化知识图谱索引器...")
        self.graph_indexer = create_graph_indexer()
        logger.info("所有组件初始化完成！")
    
    def _extract_event_info(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从事件数据中提取关键信息
        
        Args:
            event_data (Dict[str, Any]): 事件数据
            
        Returns:
            Dict[str, Any]: 提取的信息
        """
        try:
            record = event_data.get('Records', [{}])[0] #获取 Records 字段，如果没有则返回 [{}]
            s3_info = record.get('s3', {})
            '''MinIO/S3 事件数据的实际结构示例
            {
                "Records": [
                    {
                    "eventVersion": "2.1",
                    "eventSource": "minio:s3",
                    "awsRegion": "",
                    "eventTime": "2024-01-15T10:30:45.123Z",
                    "eventName": "s3:ObjectCreated:Put",
                    "userIdentity": {
                        "principalId": "minio"
                    },
                    "requestParameters": {
                        "accessKey": "your-access-key",
                        "region": "",
                        "sourceIPAddress": "127.0.0.1"
                    },
                    "responseElements": {
                        "x-amz-request-id": "123456789",
                        "x-minio-origin-endpoint": "http://localhost:9000"
                    },
                    "s3": {
                        "s3SchemaVersion": "1.0",
                        "configurationId": "Config",
                        "bucket": {
                        "name": "public-docs",#这个桶的名字是我随便起的，实际上的名字是在.env中规定好的
                        "ownerIdentity": {
                            "principalId": "minio"
                        },
                        "arn": "arn:aws:s3:::public-docs"
                        },
                        "object": {
                        "key": "documents/example.pdf",
                        "size": 1048576,
                        "eTag": "\"abc123def456\"",
                        "contentType": "application/pdf",
                        "userMetadata": {},
                        "versionId": "1",
                        "sequencer": "0000000000000001"
                        }
                    },
                    "source": {
                        "host": "localhost",
                        "port": "9000",
                        "userAgent": "MinIO (linux; amd64) minio-go/v7.0.0"
                    }
                    }
                ]
            }'''
            
            return {
                'event_name': record.get('eventName', ''),
                'event_time': record.get('eventTime', ''),
                'bucket_name': s3_info.get('bucket', {}).get('name', ''),
                'object_name': s3_info.get('object', {}).get('key', ''),
                'object_size': s3_info.get('object', {}).get('size', 0),
                'content_type': s3_info.get('object', {}).get('contentType', ''),
                'etag': s3_info.get('object', {}).get('eTag', ''),
            }
        except Exception as e:
            logger.error(f"提取事件信息失败: {e}")
            return {}
    
    def _handle_object_created(self, event_info: Dict[str, Any]):
        """
        处理对象创建事件
        
        Args:
            event_info (Dict[str, Any]): 事件信息
        """
        try:
            bucket_name = event_info['bucket_name']
            object_name = event_info['object_name']
            
            logger.info(f"\n=== 处理新增对象: {bucket_name}/{object_name} ===")
            logger.info(f"对象大小: {event_info['object_size']} 字节")
            logger.info(f"内容类型: {event_info['content_type']}")
            
            # 直接调用_process_single_object方法处理
            self._process_single_object(bucket_name, object_name, force_update=True)
            
        except Exception as e:
            logger.error(f"处理对象创建事件失败: {e}")
    
    def _handle_object_deleted(self, event_info: Dict[str, Any]):
        """
        处理对象删除事件
        
        Args:
            event_info (Dict[str, Any]): 事件信息
        """
        try:
            bucket_name = event_info['bucket_name']
            object_name = event_info['object_name']
            doc_path_name = object_name  # 不再包含bucket_name前缀
            
            logger.info(f"\n=== 处理删除对象: {bucket_name}/{object_name} ===")
            
            # 从Milvus中删除相关记录
            logger.info("从Milvus中查找并删除相关记录...")
            
            # 使用MilvusAPI的删除方法
            if self.milvus_api.delete_existing_document(doc_path_name):
                logger.info(f"成功删除文档记录: {doc_path_name}")
            else:
                logger.error(f"删除文档记录失败: {doc_path_name}")

            # 同步从知识图谱中删除
            if self.graph_indexer is not None:
                try:
                    self.graph_indexer.delete_document(doc_path_name)
                except Exception as e:
                    logger.warning(f"知识图谱删除失败（不影响向量删除）: {e}")
            
        except Exception as e:
            logger.error(f"处理对象删除事件失败: {e}")
    
    def _process_event(self, event_data: Dict[str, Any]):
        """
        处理单个事件
        
        Args:
            event_data (Dict[str, Any]): 事件数据
        """
        try:
            # 提取事件信息
            event_info = self._extract_event_info(event_data)
            if not event_info:
                logger.error("无法提取事件信息，跳过处理")
                return
            
            event_name = event_info['event_name']
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            logger.info(f"\n[{timestamp}] 收到事件: {event_name}")
            logger.info(f"对象: {event_info['bucket_name']}/{event_info['object_name']}")
            
            # 根据事件类型进行处理
            if 'ObjectCreated' in event_name:
                self._handle_object_created(event_info)
            elif 'ObjectRemoved' in event_name:
                self._handle_object_deleted(event_info)
            else:
                logger.error(f"未处理的事件类型: {event_name}")
                
        except Exception as e:
            logger.error(f"处理事件时出错: {e}")
    
    def _process_single_object(self, bucket_name: str, object_name: str, force_update: bool = False):
        """
        处理单个对象（用于全量刷新和排查补漏）
        对象指的是bucket桶里面的文件对象
        
        Args:
            bucket_name (str): 桶名称
            object_name (str): 对象名称
            force_update (bool): 是否强制更新（True=全量刷新，False=排查补漏）
        
        Returns:
            bool: 处理是否成功
        """
        try:
            doc_path_name = object_name  # 不再包含bucket_name前缀
            
            # 如果是排查补漏模式，先检查是否已存在。系统第一次运行执行全量刷新模式，后续都只执行排查不漏模式
            if not force_update:
                if self.milvus_api.check_document_exists(doc_path_name):
                    print(f"  文档已存在与milvus中，跳过: {object_name}")
                    return True
            
            logger.info(f"  处理文档: {object_name}")
            
            # 解析对象内容
            parse_result = self.parser.parse_object(bucket_name, object_name)
            '''
            parse_result数据示例：
            {
                "doc_name": "example.pdf",                    # 文档名称（文件名）
                "doc_path_name": "documents/example.pdf",     # 文档路径（在MinIO中的完整路径）
                "doc_type": "pdf",                            # 文档类型（pdf, docx, txt, md等）
                "doc_md5": "abc123def45678901234567890abcdef", # 文档MD5哈希值（32位十六进制字符串）
                "doc_length": 1048576,                        # 文档大小（字节数），这里是1MB
                "contents": [                                 # 文本片段列表（字符串列表）
                    "这是第一个文本片段的内容。文档的开头部分...",
                    "...片段之间有128个字符的重叠...这是第二个片段...",
                    "...这是第三个片段...",
                    "...文档的最后部分..."
                ]
            }
            '''
            
            if 'error' in parse_result:
                logger.error(f"    解析失败: {parse_result['error']}")
                return False
            
            if not parse_result['contents']:
                logger.error(f"    未提取到文本内容，跳过")
                return True
            
            logger.info(f"    解析成功，获得 {len(parse_result['contents'])} 个文本片段")
            
            # 如果是强制更新，先删除已存在的记录
            if force_update:
                self.milvus_api.delete_existing_document(doc_path_name)
            
            # 为每个文本片段生成embedding并存储到Milvus
            success_count = 0
            for i, content in enumerate(parse_result['contents']):
                try:
                    # 生成embedding向量
                    embedding_vector = self.embedding_tool.get_embedding(content)
                    if not embedding_vector:
                        logger.error(f"    片段 {i+1} embedding生成失败，跳过")
                        continue
                    
                    # 准备文档数据
                    doc_data = MilvusDocument(
                        doc_name=parse_result['doc_name'],
                        doc_path_name=parse_result['doc_path_name'],
                        doc_type=parse_result['doc_type'],
                        doc_md5=parse_result['doc_md5'],
                        doc_length=parse_result['doc_length'],
                        content=content,
                        content_vector=embedding_vector,
                        embedding_model=self.embedding_tool.model
                    )
                    
                    # 存储到Milvus（允许重复，因为我们已经处理了去重逻辑）
                    if self.milvus_api.insert_document(doc_data):
                        success_count += 1
                    else:
                        logger.error(f"    片段 {i+1} 存储失败")
                
                except Exception as e:
                    logger.error(f"    处理片段 {i+1} 时出错: {e}")
                    continue
            
            logger.info(f"    完成！成功存储 {success_count}/{len(parse_result['contents'])} 个片段")

            # 同步导入知识图谱（如果图谱索引器可用）
            if self.graph_indexer is not None:
                try:
                    graph_result = self.graph_indexer.import_document(
                        doc_name=parse_result['doc_name'],
                        doc_path_name=parse_result['doc_path_name'],
                        text_chunks=parse_result['contents'],
                    )
                    logger.info(
                        "    图谱导入: %d 实体, %d 关系",
                        graph_result['entities_count'],
                        graph_result['relations_count'],
                    )
                except Exception as e:
                    logger.warning(f"    知识图谱导入失败（不影响向量索引）: {e}")

            return success_count > 0
            
        except Exception as e:
            logger.error(f"  处理对象失败: {e}")
            return False
    
    def full_update(self):
        """
        模式1：全量刷新 - 遍历Minio桶中的所有数据并更新到Milvus
            对每个对象调用 _process_single_object(force_update=True)
            先删除旧数据，再重新解析、重新写入 Milvus
            适用：首次构建索引，或需要强制重建全部向量的场景
        """
        logger.info(f"\n=== 模式1：全量刷新 ===")
        logger.info(f"正在遍历桶 '{self.bucket_name}' 中的所有对象...")
        
        try:
            # 获取桶中的所有对象
            objects = self.minio_client.list_objects(self.bucket_name, recursive=True) #循环处理
            
            total_objects = 0
            processed_objects = 0
            
            for obj in objects:
                total_objects += 1
                object_name = obj.object_name
                
                # 跳过文件夹
                if object_name.endswith('/'):
                    continue
                
                logger.info(f"\n[{total_objects}] 处理对象: {object_name}")
                
                if self._process_single_object(self.bucket_name, object_name, force_update=True):
                    processed_objects += 1
                    
            self.milvus_api.flush_collection()
            
            logger.info(f"\n=== 全量刷新完成 ===")
            logger.info(f"总对象数: {total_objects}")
            logger.info(f"成功处理: {processed_objects}")
            logger.info(f"失败数量: {total_objects - processed_objects}")
            
        except Exception as e:
            logger.error(f"全量刷新失败: {e}")
    
    def backfill_update(self):
        """
        模式2：排查补漏 - 检查Milvus中不存在的文档并新增
        """
        logger.info(f"\n=== 模式2：排查补漏 ===")
        logger.info(f"正在检查桶 '{self.bucket_name}' 中缺失的文档...")
        
        try:
            # 获取桶中的所有对象
            objects = self.minio_client.list_objects(self.bucket_name, recursive=True)
            
            total_objects = 0
            new_objects = 0
            existing_objects = 0
            
            for obj in objects:
                total_objects += 1
                object_name = obj.object_name
                
                # 跳过文件夹
                if object_name.endswith('/'):
                    continue
                
                logger.info(f"\n[{total_objects}] 检查对象: {object_name}")
                
                # 检查是否已存在
                if self.milvus_api.check_document_exists(object_name):
                    logger.info(f"  已存在，跳过")
                    existing_objects += 1
                else:
                    logger.info(f"  不存在，开始处理...")
                    if self._process_single_object(self.bucket_name, object_name, force_update=False):
                        new_objects += 1
            
            self.milvus_api.flush_collection()
            
            logger.info(f"\n=== 排查补漏完成 ===")
            logger.info(f"总对象数: {total_objects}")
            logger.info(f"已存在: {existing_objects}")
            logger.info(f"新增: {new_objects}")
            logger.info(f"失败数量: {total_objects - existing_objects - new_objects}")
            
        except Exception as e:
            logger.error(f"排查补漏失败: {e}")
    
    def start_listening(self):
        """
        模式3：增量更新 - 根据消息通知进行增量更新
        """
        logger.info(f"\n=== 模式3：增量更新 ===")
        logger.info(f"开始监听桶 '{self.bucket_name}' 的事件...")
        logger.info("按 Ctrl+C 停止监听")
        
        try:
            # 监听桶事件
            #当minio中的test这个桶中有数据上传或者删除的时候，这个监听被触发
            events = self.minio_client.listen_bucket_notification(
                bucket_name=self.bucket_name,
                events=['s3:ObjectCreated:*', 's3:ObjectRemoved:*']
            )
            
            for event in events:
                try:
                    #1. 将 MinIO 事件监听器返回的不同格式的事件数据统一转换为字典（dict）格式
                    #因为MinIO 的 listen_bucket_notification 可能返回不同格式的事件数据（不同版本或配置下可能是 bytes、字符串或字典）
                    if event:
                        # 解析事件数据
                        if isinstance(event, bytes):
                            event_data = json.loads(event.decode('utf-8'))
                        elif isinstance(event, str):
                            event_data = json.loads(event)
                        elif isinstance(event, dict):
                            event_data = event
                        else:
                            logger.error(f"未知的事件数据类型: {type(event)}")
                            continue
                        
                        # 2. 处理事件
                        self._process_event(event_data)
                        
                except json.JSONDecodeError as e:
                    logger.error(f"解析事件数据失败: {e}")
                except Exception as e:
                    logger.error(f"处理事件失败: {e}")
                    
        except KeyboardInterrupt:
            logger.info("\n监听已停止")
        except Exception as e:
            logger.error(f"监听过程中出错: {e}")
    


def main():
    """主函数"""
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="Minio文档处理系统 - 将Minio中的文档解析并存储到Milvus向量数据库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                    使用示例:
                    # 全量刷新模式
                    python main.py --mode full-refresh
                    
                    # 排查补漏模式
                    python main.py --mode backfill
                    
                    # 增量更新模式（实时监听）
                    python main.py --mode listen
                    
                    # 后台运行增量更新
                    nohup python main.py --mode listen > minio_listener.log 2>&1 &
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["full-refresh", "backfill", "listen"],
        required=True,
        help="运行模式选择: full-refresh=全量刷新, backfill=排查补漏, listen=增量更新(实时监听)"
    )
    
    parser.add_argument(
        "--bucket",
        type=str,
        help="指定要处理的Minio桶名称，默认从环境变量MINIO_BUCKET获取"
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    logger.info("=== Minio文档处理系统 ===")
    logger.info(f"运行模式: {args.mode}")
    
    try:
        # 创建监听器实例
        if args.bucket:
            listener = MinioEventListener(bucket_name=args.bucket)
            logger.info(f"使用指定桶: {args.bucket}")
        else:
            listener = MinioEventListener()
            logger.info(f"使用默认桶: {listener.bucket_name}")
        
        logger.info("=== 系统初始化完成 ===")
        
        # 根据模式执行相应操作
        if args.mode == "full-refresh":
            logger.info("\n执行全量刷新模式...")
            listener.full_update()
            
        elif args.mode == "backfill":
            logger.info("\n执行排查补漏模式...")
            listener.backfill_update()
            
        elif args.mode == "listen":
            logger.info("\n执行增量更新模式（实时监听）...")
            logger.info("提示: 使用 Ctrl+C 停止监听，或使用 nohup 在后台运行")
            listener.start_listening()
        else:
            ## 使用方式说明
            logger.info("""
            使用示例:
            # 全量刷新模式
            python main.py --mode full-refresh

            # 排查补漏模式
            python main.py --mode backfill

            # 增量更新模式（实时监听）
            python main.py --mode listen            
            """)


        logger.info("\n程序执行完成")
        
    except KeyboardInterrupt:
        logger.info("\n用户中断程序")
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        exit(1)


if __name__ == "__main__":
    main()
