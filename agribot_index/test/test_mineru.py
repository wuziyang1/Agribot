
from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, read_fn
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
from mineru.utils.enum_class import MakeMode

pdf_file_name = "data/pdf3.pdf" # 示例文件名

print(f"Converting {pdf_file_name} to bytes")

pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(read_fn(pdf_file_name), 0, None)

print(f"Converted {pdf_file_name} to bytes")

print(f"Analyzing {pdf_file_name}")
middle_json, infer_result = vlm_doc_analyze(pdf_bytes, backend="sglang-client", server_url="http://127.0.0.1:30000", image_writer=None)

print(f"Analyzed {pdf_file_name}")

pdf_info = middle_json["pdf_info"]

print(f"Making markdown content")
md_content_str = vlm_union_make(pdf_info, MakeMode.MM_MD, None)

print(f"Made markdown content")

print(md_content_str)

print(f"Writing markdown content to file")

with open("pdf3.md", "w") as f:
    f.write(md_content_str)

print(f"Wrote markdown content to file")



