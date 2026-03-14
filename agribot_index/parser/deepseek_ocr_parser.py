import base64
import os
import sys
import re
from typing import List

import fitz  # PyMuPDF，用于 PDF 转图片供 OCR

from dotenv import load_dotenv
from openai import OpenAI

# 确保能找到 parser 和 logger 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser.document_parser import DocumentParser
from logger.logging import setup_logging

load_dotenv()
logger = setup_logging()

class PDFOCRParser(DocumentParser):
    def __init__(self):
        # 优先读取 OCR 专用配置，没有则继承通用 LLM 配置
        self.api_key = os.getenv("OCR_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OCR_BASE_URL") or os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        self.model = os.getenv("OCR_MODEL_NAME", "PaddlePaddle/PaddleOCR-VL-1.5")

        # OCR 渲染分辨率，固定为 200 DPI（不再从环境变量读取）
        self.dpi = 200
        # OCR_MAX_PAGES <= 0 表示不限制页数，默认对整本 PDF 做 OCR
        self.max_pages = int(os.getenv("OCR_MAX_PAGES", "0"))
        # 单页 OCR 超时时间（秒），固定为 120 秒（不再从环境变量读取）
        self.timeout_s = 120.0
        self.retry_dpi_list = self._parse_dpi_list(os.getenv("OCR_RETRY_DPI_LIST", ""))
        self.min_chars_per_page = int(os.getenv("OCR_MIN_CHARS_PER_PAGE", "30"))
        self.max_junk_ratio = float(os.getenv("OCR_MAX_JUNK_RATIO", "0.65"))

        if not self.api_key or not self.base_url:
            raise ValueError("OCR API 配置缺失")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def supports(self, content_type: str) -> bool:
        return content_type.lower() in ["application/pdf", "pdf"]

    def parse(self, data: bytes) -> str:
        if os.getenv("ENABLE_PDF_OCR", "true").lower() != "true":
            return ""

        try:
            doc = fitz.open(stream=data, filetype="pdf")
            try:
                total_pages = len(doc)
                if total_pages <= 0: return ""
                
                page_limit = total_pages if self.max_pages <= 0 else min(total_pages, self.max_pages)
                dpi_candidates = self.retry_dpi_list or [self.dpi]
                final_texts = []

                for idx in range(page_limit):
                    current_page_text = ""
                    page_no = idx + 1
                    
                    # 逐页尝试不同的 DPI 直到质量达标
                    for attempt, dpi in enumerate(dpi_candidates, start=1):
                        try:
                            img_bytes = self._render_page(doc, idx, dpi)
                            candidate = self._ocr_request(img_bytes, page_no, strict=(attempt > 1))
                            
                            if self._looks_good(candidate):
                                current_page_text = candidate
                                break
                            current_page_text = candidate or current_page_text
                            logger.info(f"Page {page_no} quality low, retrying with DPI {dpi}")
                        except Exception as e:
                            logger.error(f"Page {page_no} OCR attempt {attempt} failed: {e}")
                            continue
                    
                    if current_page_text:
                        final_texts.append(current_page_text.strip())

                return "\n\n".join(final_texts).strip()
            finally:
                doc.close()
        except Exception as e:
            logger.error(f"CRITICAL: PDF OCR Parser failed: {e}")
            return ""

    def _render_page(self, doc, idx: int, dpi: int) -> bytes:
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = doc.load_page(idx).get_pixmap(matrix=matrix, alpha=False)
        # 使用 jpeg 减少传输体积，防止 API 报 413 Payload Too Large
        return pix.tobytes("jpeg")

    def _ocr_request(self, img_bytes: bytes, page_no: int, strict: bool) -> str:
        b64_data = base64.b64encode(img_bytes).decode("utf-8")

        # 参考官方 transformers 示例：图片在前，任务关键词 "OCR:" 在后
        # https://ernie.baidu.com/blog/zh/posts/paddleocr-vl-1.5/
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}},
                    {"type": "text", "text": "OCR:"}
                ]
            }],
            temperature=0.0,
            timeout=self.timeout_s
        )

        raw_content = (response.choices[0].message.content or "").strip()
        if not raw_content:
            return ""
        return f"[第{page_no}页]\n{raw_content}"

    def _parse_dpi_list(self, v: str) -> List[int]:
        if not v: return []
        try:
            nums = [int(p.strip()) for p in v.split(",") if p.strip()]
            return list(dict.fromkeys(nums))
        except: return []

    def _looks_good(self, text: str) -> bool:
        if not text: return False
        body = re.sub(r"^\[第\d+页\]\s*", "", text)
        if len(body) < self.min_chars_per_page: return False
        
        # 扩充后的合法字符集（含常用数学符号和货币符号）
        legal_pattern = r"[\u4e00-\u9fffA-Za-z0-9\s，。；：、！？（）()《》“”‘’【】\[\]{}<>·…—\-_,.;:!?/\\\"'\n\r\t@#$%^&*+=~|￥€£]"
        found_chars = re.findall(legal_pattern, body)
        junk_ratio = 1.0 - (len(found_chars) / len(body))
        return junk_ratio <= self.max_junk_ratio