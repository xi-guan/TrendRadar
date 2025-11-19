"""
LangChain Chains

实现各种处理链：
- 智能摘要链
- RAG 链 (检索增强生成)
- 对话式 RAG 链
"""

from langchain_agents.chains.summary_chain import (
    NewsSummaryChain,
    MultipleNewsSummaryChain,
    TrendAnalysisSummaryChain,
    create_summary_chain,
)
from langchain_agents.chains.rag_chain import (
    RAGChain,
    ConversationalRAGChain,
    create_rag_chain,
)

__all__ = [
    "NewsSummaryChain",
    "MultipleNewsSummaryChain",
    "TrendAnalysisSummaryChain",
    "create_summary_chain",
    "RAGChain",
    "ConversationalRAGChain",
    "create_rag_chain",
]
