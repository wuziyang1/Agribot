from abc import ABC, abstractmethod

'''
他的作用就是
    让下面具体的解析器继承并实现这两个方法
'''

class DocumentParser(ABC):
    """文档解析器抽象基类
        ABC是抽象基类，用于定义抽象方法，子类必须实现这些方法
        
    继承关系为：
        DocumentParser (抽象基类)
            ├── PDFParser (PDF解析器)
            ├── OfficeParser (Office文档解析器)
            ├── MarkdownParser (Markdown解析器)
            ├── TextParser (文本解析器)
            └── MinerUParser (MinerU解析器)
    """
    
    @abstractmethod
    def parse(self, data: bytes) -> str:
        """
        解析文档内容
        
        Args:
            data (bytes): 文档二进制数据
            
        Returns:
            str: 解析出的文本内容
        """
        pass
    
    @abstractmethod
    def supports(self, content_type: str) -> bool:
        """
        检查是否支持指定的内容类型
        
        Args:
            content_type (str): 内容类型
            
        Returns:
            bool: 是否支持
        """
        pass