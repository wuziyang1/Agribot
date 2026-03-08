from minio import Minio
from dotenv import load_dotenv
import os
from pprint import pprint
import PyPDF2
from io import BytesIO

load_dotenv()

client = Minio(
    endpoint=os.getenv("ENDPOINT"),
    access_key=os.getenv("ACCESS_KEY"),
    secret_key=os.getenv("SECRET_KEY"),
    secure=False
)

resp = client.get_object("test", "demo/GoogleIO.pdf")

pprint(resp.status)
pprint(resp.headers)
pprint(resp.headers.get("Content-Type"))
pprint(resp.headers.get("Content-Length"))
pprint(resp.headers.get("ETag"))


reader = PyPDF2.PdfReader(BytesIO(resp.data))

num_pages = len(reader.pages)

text_content = ""
for page_num in range(num_pages):
    page = reader.pages[page_num]
    page_text = page.extract_text()
    if page_text:
        text_content += page_text + "\n"

if text_content.strip():
    print(f"    ✅ 提取文本: {len(text_content)} 字符")
else:
    print("    ❌ 未提取到文本内容")

print(text_content)