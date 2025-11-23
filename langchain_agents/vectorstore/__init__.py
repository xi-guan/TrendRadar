"""
Vector Store 模块

提供向量存储和检索功能：
- Chroma: 向量数据库集成
- NewsVectorStore: 新闻文档的向量化存储和检索
"""

from langchain_agents.vectorstore.chroma_store import (
    NewsVectorStore,
    create_news_vectorstore,
    get_default_vectorstore,
)

__all__ = [
    "NewsVectorStore",
    "create_news_vectorstore",
    "get_default_vectorstore",
]
