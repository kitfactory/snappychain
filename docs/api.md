
# Chat


|関数|説明|
|dev|チェーンを開発モードで始動します|

| 関数 | 説明 |
|---|---|
|chat_openai(model: str = "gpt-4o-mini", temperature: float = 0.2)| OpenAIのチャットモデルを使用する。必要な引数は、modelとtemperatureとして指定されます。 |
|chat_ollama(model: str = "llama2", temperature: float = 0.2)| Ollamaのチャットモデルを使用する |
|chat_gemini()| Geminiのチャットモデルを使用する |
|chat_anthropic()| Anthropicのチャットモデルを使用する |


# schema.py functions89;[0\]
'schema',

# prompt.py functions
'system_prompt',
'human_prompt',
'ai_prompt',

# output.py functions
'output',

# loader.py functions
'text_load',
'pypdf_load',
'markitdown_load',
'directory_load',       
'unstructured_markdown_load',

'get_chain_documents',　## ?

# splitters.py functions,
'split_text',
'recursive_split_text',
'markdown_text_splitter',
'python_text_splitter',
'json_text_splitter',

# vectorstore.py functions
'add_documents_to_vector_store',
'persist_vector_store',
'query_vector_store',

    'faiss_vs_query',
    'faiss_vs_store',
    'chroma_vs_query',
    'chroma_vs_store',

    # embedding.py functions
    'openai_embedding',
    'ollama_embedding',

    # wikipedia.py functions
    'wikipedia_to_text',

    # rag.py functions
    'build_rag_chain',

    # vectorstore.py functions
    'UnifiedVectorStore',

    # rerank.py functions
    'UnifiedRerank',


    # bm25sj.py functions
    'BM25SJRetriever',
    'bm25sj',
    'bm25sj_query',

    # epistemize.py functions
    'epistemize',
