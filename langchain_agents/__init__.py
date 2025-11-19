"""
TrendRadar LangChain 1.0 集成

提供 AI 增强功能：
- 智能新闻摘要
- 趋势预测分析
- 语义搜索
- 对话式交互

这是可选功能，需要安装 langchain 依赖组：
    uv sync --group langchain
"""

__version__ = "1.0.0"

from .config import LangChainConfig

__all__ = ["LangChainConfig"]
