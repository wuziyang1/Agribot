from openai import OpenAI

# 注意：因为有隧道，base_url 直接连 localhost
client = OpenAI(
    api_key="vllm-no-key-needed",
    base_url="http://127.0.0.1:8000/v1"
)

texts = ["如何评价 Gemini 3 Flash 的性能？", "深度学习在医疗领域的应用"]

responses = client.embeddings.create(
    input=texts,
    model="bge-m3"
)

for i, data in enumerate(responses.data):
    print(f"文本 {i} 的向量维度: {len(data.embedding)}")
    print(f"前 5 维数据: {data.embedding[:5]}")