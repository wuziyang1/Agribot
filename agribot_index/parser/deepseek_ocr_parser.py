import base64
from io import BytesIO
import os
import sys
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser.document_parser import DocumentParser
from logger.logging import setup_logging

load_dotenv()
logger = setup_logging()


class DeepSeekOCRParser(DocumentParser):
    """
    使用 deepseek-ai/DeepSeek-OCR 对扫描件 PDF 做 OCR。

    调用的是 OpenAI-compatible 的 chat.completions，并把每页渲染成 PNG 后以 data URL 形式传入。
    """

    def __init__(self):
        self.api_key = os.getenv("OCR_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OCR_BASE_URL") or os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        self.model = os.getenv("OCR_MODEL_NAME", "deepseek-ai/DeepSeek-OCR")

        # 渲染/调用参数
        self.dpi = int(os.getenv("OCR_PDF_DPI", "200"))
        self.max_pages = int(os.getenv("OCR_MAX_PAGES", "20"))
        self.timeout_s = float(os.getenv("OCR_TIMEOUT_S", "120"))

        if not self.api_key or not self.base_url:
            raise ValueError("OCR_API_KEY / OCR_BASE_URL 未配置（或无法从 LLM_/OPENAI_ 变量继承）")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def supports(self, content_type: str) -> bool:
        return content_type.lower() in ["application/pdf", "pdf"]

    def parse(self, data: bytes) -> str:
        enabled = os.getenv("ENABLE_PDF_OCR", "true").lower() == "true"
        if not enabled:
            return ""

        try:
            pages_png = self._render_pdf_to_png_pages(data)
            if not pages_png:
                return ""

            texts: List[str] = []
            for idx, png_bytes in enumerate(pages_png, start=1):
                try:
                    page_text = self._ocr_png(png_bytes, page_index=idx)
                    if page_text:
                        texts.append(page_text.strip())
                except Exception as e:
                    logger.error(f"DeepSeek-OCR 第{idx}页识别失败: {e}")
                    continue

            return "\n\n".join([t for t in texts if t]).strip()
        except Exception as e:
            logger.error(f"DeepSeek-OCR 解析失败: {e}")
            return ""

    def _render_pdf_to_png_pages(self, pdf_bytes: bytes) -> List[bytes]:
        try:
            import fitz  # PyMuPDF
        except Exception as e:
            raise ImportError("缺少依赖 PyMuPDF（pymupdf），无法将PDF渲染为图片用于OCR") from e

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            page_count = min(len(doc), self.max_pages)
            if page_count <= 0:
                return []

            zoom = self.dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)

            pages: List[bytes] = []
            for i in range(page_count):
                page = doc.load_page(i)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                pages.append(pix.tobytes("png"))
            return pages
        finally:
            doc.close()

    def _ocr_png(self, png_bytes: bytes, page_index: int) -> str:
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"

        prompt = (
            "请对图片进行OCR识别，尽可能完整地输出所有可见文字内容。"
            "保持原有的段落与换行；如果有表格，请按行输出；不要添加任何额外解释。"
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            max_tokens=4092,
            timeout=self.timeout_s,
        )

        if not resp.choices:
            return ""
        msg = resp.choices[0].message
        content = (msg.content or "").strip()
        if not content:
            return ""
        return f"[第{page_index}页]\n{content}"

