"""
LangChain Chains

实现各种处理链：
- 智能摘要链
- 分析链
- 对比分析链
"""

from langchain_agents.chains.summary_chain import (
    NewsSummaryChain,
    MultipleNewsSummaryChain,
    TrendAnalysisSummaryChain,
    create_summary_chain,
)

__all__ = [
    "NewsSummaryChain",
    "MultipleNewsSummaryChain",
    "TrendAnalysisSummaryChain",
    "create_summary_chain",
]
