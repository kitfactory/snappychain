from .chat import openai_chat   #, ollama_chat  # , gemini_chat, ollama_chat, anthropic_chat
from .prompt import system_prompt, human_prompt, ai_prompt
from .schema import schema
from .devmode import dev
from .output import output
from .embedding import add_documents_to_vector_store, persist_vector_store, query_vector_store, openai_embedding, ollama_embedding

from .loader import (
    text_load,
    pypdf_load,
    markitdown_load,
    get_chain_documents,
    directory_load,
    unstructured_markdown_load,
)

from .splitter import (
    split_text,
    recursive_split_text,
    markdown_text_splitter,
    python_text_splitter,
    json_text_splitter
)
from .wikipedia import wikipedia_to_text

__all__ = [
    # chat.py functions
    'openai_chat',
    # 'ollama_chat', 
    # 'gemini_chat',
    # 'anthropic_chat',

    # devmode.py functions
    'dev',
    # schema.py functions
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
    'get_chain_documents',
    'directory_load',
    'unstructured_markdown_load',

    # splitters.py functions
    'split_text',
    'recursive_split_text',
    'markdown_text_splitter',
    'python_text_splitter',
    'json_text_splitter',

    # vectorstore.py functions
    'add_documents_to_vector_store',
    'persist_vector_store',
    'query_vector_store',
    
    # embedding.py functions
    'openai_embedding',
    'ollama_embedding',

    # wikipedia.py functions
    'wikipedia_to_text'

    # rag utilities
    # 'to_documents',
    # 'rag_setup',
    # 'rag_query',
    # 'rag_add_documents',
]
