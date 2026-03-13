import base64
from io import BytesIO
import os
import sys
import re
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
        # 注意：扫描件 OCR 识别率和 DPI 强相关。这里支持按页自动重试多个 DPI。
        self.dpi = int(os.getenv("OCR_PDF_DPI", "200"))
        self.max_pages = int(os.getenv("OCR_MAX_PAGES", "200"))
        self.timeout_s = float(os.getenv("OCR_TIMEOUT_S", "120"))
        self.retry_dpi_list = self._parse_dpi_list(os.getenv("OCR_RETRY_DPI_LIST", ""))
        self.min_chars_per_page = int(os.getenv("OCR_MIN_CHARS_PER_PAGE", "30"))
        self.max_junk_ratio = float(os.getenv("OCR_MAX_JUNK_RATIO", "0.65"))

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
            try:
                import fitz  # PyMuPDF
            except Exception as e:
                raise ImportError("缺少依赖 PyMuPDF（pymupdf），无法将PDF渲染为图片用于OCR") from e

            doc = fitz.open(stream=data, filetype="pdf")
            try:
                total_pages = len(doc)
                if total_pages <= 0:
                    return ""

                # max_pages <= 0 表示不限页（整本识别，注意成本/耗时）
                page_count = total_pages if self.max_pages <= 0 else min(total_pages, self.max_pages)

                # 每页 DPI 尝试列表：优先用 retry 列表；否则使用单个 dpi
                dpi_candidates = self.retry_dpi_list or [self.dpi]

                texts: List[str] = []
                for page_idx_0 in range(page_count):
                    page_no = page_idx_0 + 1
                    page_text = ""

                    for attempt, dpi in enumerate(dpi_candidates, start=1):
                        png_bytes = self._render_pdf_page_to_png(doc, page_idx_0, dpi=dpi)
                        candidate = self._ocr_png(
                            png_bytes,
                            page_index=page_no,
                            strict_text_only=(attempt > 1),
                        )

                        # 质量评估：太短 or 垃圾字符占比过高 → 继续重试更高 DPI
                        if self._looks_good(candidate):
                            page_text = candidate
                            break

                        page_text = candidate or page_text
                        logger.info(
                            f"DeepSeek-OCR 第{page_no}页质量不足，尝试重试: attempt={attempt} dpi={dpi}"
                        )

                    if page_text:
                        texts.append(page_text.strip())

                return "\n\n".join([t for t in texts if t]).strip()
            finally:
                doc.close()
        except Exception as e:
            logger.error(f"DeepSeek-OCR 解析失败: {e}")
            return ""

    def _render_pdf_page_to_png(self, doc, page_index_0: int, dpi: int) -> bytes:
        zoom = dpi / 72.0
        matrix = None
        try:
            import fitz  # type: ignore

            matrix = fitz.Matrix(zoom, zoom)
        except Exception:
            # 理论上不会发生（doc 已打开），这里只是兜底
            matrix = None

        page = doc.load_page(page_index_0)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        return pix.tobytes("png")

    def _ocr_png(self, png_bytes: bytes, page_index: int, strict_text_only: bool = False) -> str:
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"

        # 第二次及以上重试时，尽量约束为“只输出可见字符文本”，减少模型自发 LaTeX/扩写。
        if strict_text_only:
            prompt = (
                "请对图片进行OCR识别，尽可能完整地逐字输出所有可见文字。"
                "只输出文本内容本身，不要解释、不要总结、不要补全缺失内容。"
                "保持原有段落与换行；表格按行输出。"
                "如果存在数学公式/符号，请按图片中可见字符原样输出，不要将其改写为LaTeX。"
            )
        else:
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

    def _parse_dpi_list(self, v: str) -> List[int]:
        if not v:
            return []
        parts = [p.strip() for p in v.split(",") if p.strip()]
        out: List[int] = []
        for p in parts:
            try:
                n = int(p)
                if n > 0:
                    out.append(n)
            except Exception:
                continue
        # 去重但保序
        seen = set()
        uniq: List[int] = []
        for n in out:
            if n in seen:
                continue
            uniq.append(n)
            seen.add(n)
        return uniq

    def _looks_good(self, page_text: str) -> bool:
        if not page_text:
            return False
        # 去掉页头标记后再评估
        body = re.sub(r"^\[第\d+页\]\s*", "", page_text.strip())
        if len(body) < self.min_chars_per_page:
            return False
        # 垃圾字符占比：非中英文/数字/常见标点/空白的比例过高，认为识别质量差
        # 这里允许常见中文标点和基本符号，减少误伤正常文本
        allowed = re.compile(r"[\u4e00-\u9fffA-Za-z0-9\s，。；：、！？（）()《》“”‘’【】\[\]{}<>·…—\-_,.;:!?/\\\"'\n\r\t]")
        allowed_count = len(allowed.findall(body))
        total = max(len(body), 1)
        junk_ratio = 1.0 - (allowed_count / total)
        return junk_ratio <= self.max_junk_ratio

