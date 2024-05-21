import tensorflow_hub as hub

# 加载 Universal Sentence Encoder 模型
model = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4")

# 生成嵌入
embeddings = model([
    'I am a sentence for which I would like to get its embedding.',
])

print(embeddings)