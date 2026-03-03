#!/usr/bin/env python3
"""
查询Milvus中的数据
该测试文件用于测试 Milvus 向量数据库的查询功能
"""

# 使用相对导入
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from milvus_api import MilvusAPI
from embedding import EmbeddingTool

def query_documents():
    """查询Milvus中的文档数据"""
    print("=== 查看Milvus中所有已经存在的文档数据 ===")
    
    try:
        # 初始化API
        milvus_api = MilvusAPI()
        embedding_tool = EmbeddingTool()
        
        # 获取集合信息
        collection_info = milvus_api.get_collection_info()
        if collection_info:
            print(f"集合名称: {collection_info['collection_name']}")
            print(f"字段数量: {len(collection_info['fields'])}")
        
        # 执行简单查询来获取所有文档
        try:
            # 查询所有文档的基本信息
            results = milvus_api.client.query(
                collection_name=milvus_api.collection_name,
                filter="",  # 空表达式表示查询所有
                output_fields=["id", "doc_name", "doc_path_name", "doc_type", "doc_md5", "doc_length", "embedding_model", "content", "content_vector"],
                limit=100
            )
            
            print(f"\n找到 {len(results)} 条记录:")
            for i, doc in enumerate(results):
                print(f"\n文档 {i+1}:")
                print(f"  名称: {doc.get('doc_name', 'N/A')}")
                print(f"  路径: {doc.get('doc_path_name', 'N/A')}")
                print(f"  类型: {doc.get('doc_type', 'N/A')}")
                print(f"  MD5: {doc.get('doc_md5', 'N/A')}")
                print(f"  大小: {doc.get('doc_length', 'N/A')} 字节")
                print(f"  模型: {doc.get('embedding_model', 'N/A')}")
                print(f"  内容: {doc.get('content', 'N/A')[:100]}...")
                print(f"  内容向量: {doc.get('content_vector', 'N/A')[:10]}...")
        except Exception as e:
            print(f"查询失败: {e}")        
        
        #test_query(milvus_api, embedding_tool)

    except Exception as e:
        print(f"❌ 查询测试失败: {e}")

def test_query():        
        # 测试相似性搜索
        print(f"\n=== 测试相似性搜索 ===")
        test_query = "Gemma3 端侧部署" #搜索milvus中和这句话相似的文档
        print(f"查询文本: {test_query}")
        
        milvus_api = MilvusAPI()
        embedding_tool = EmbeddingTool()
        
        # 生成查询向量
        query_vector = embedding_tool.get_embedding(test_query)
        if query_vector:
            print(f"查询向量维度: {len(query_vector)}")
            
            # 执行相似性搜索
            search_results = milvus_api.search_similar_documents(query_vector, limit=3)
            
            print(f"找到 {len(search_results)} 个相似文档:")
            for i, result in enumerate(search_results):
                print(f"\n相似文档 {i+1}:")
                print(f"  相似度分数: {result.get('distance', 'N/A')}")
                print(f"  文档名称: {result.get('entity', {}).get('doc_name', 'N/A')}")
                print(f"  文档路径: {result.get('entity', {}).get('doc_path_name', 'N/A')}")
                print(f"  内容预览: {result.get('entity', {}).get('content', 'N/A')[:100]}...")
        
        print(f"\n✅ 查询测试完成！")

if __name__ == "__main__":
    # test_query()
    query_documents()