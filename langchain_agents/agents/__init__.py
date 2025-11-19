"""
LangChain Agents

实现各种 AI Agent：
- NewsAnalystAgent: 新闻分析 Agent
- TrendPredictorAgent: 趋势预测 Agent (LangGraph)
- NewsQAAgent: 对话式问答 Agent
"""

from langchain_agents.agents.news_analyst import NewsAnalystAgent
from langchain_agents.agents.trend_predictor import TrendPredictorAgent

__all__ = [
    "NewsAnalystAgent",
    "TrendPredictorAgent",
]
