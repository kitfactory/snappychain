
# BM25S-jライブラリの使用方法

```python
import bm25s

# ほむらの情報を含むコーパス例
corpus = [
    "暁美 ほむら（あけみ ほむら）は、テレビアニメ『魔法少女まどか☆マギカ』に登場する架空の人物です。",
    "ほむらはタイムリープを繰り返し、悲劇的な過去を持つ魔法少女です。",
    "物語の中で、ほむらは大切な使命を帯び、運命に立ち向かいます。"
]

# 日本語用のトークナイザーにより、コーパスをトークン化（stopwordsに"japanese"を指定）
corpus_tokens = bm25s.tokenize(corpus, stopwords="japanese")
print("コーパスのトークン:", corpus_tokens)

# BM25 インスタンスを作成し、コーパスのトークン化結果からインデックスを構築
retriever = bm25s.BM25()
retriever.index(corpus_tokens)

# クエリ例：「ほむらは誰？」
query = "ほむらは誰？"
query_tokens = bm25s.tokenize(query, stopwords="japanese")
print("クエリのトークン:", query_tokens)

# 検索を実行し、上位2件の結果を取得
results, scores = retriever.retrieve(query_tokens, corpus=corpus, k=2)
for i in range(results.shape[1]):
    doc, score = results[0, i], scores[0, i]
    print(f"Rank {i+1} (score: {score:.2f}): {doc}")

```
