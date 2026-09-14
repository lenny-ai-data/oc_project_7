"""Vérifie que l'environnement s'importe bien

Usage : uv run python scripts/check_env.py
"""

import faiss
from langchain_community.vectorstores import FAISS
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

print(f"faiss {faiss.__version__} OK")
print(f"langchain-mistralai OK ({MistralAIEmbeddings.__name__}, {ChatMistralAI.__name__})")