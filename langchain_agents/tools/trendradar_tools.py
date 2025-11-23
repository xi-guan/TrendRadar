"""
TrendRadar Tools 包装器

将 TrendRadar 现有的 MCP 工具包装为 LangChain Tools，
使得 LangChain Agent 可以调用这些工具。
"""

import asyncio
import json
import logging
from typing import List, Optional, Dict, Any, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from mcp_server.tools.data_query import DataQueryTools
from mcp_server.tools.analytics import AnalyticsTools
from mcp_server.tools.search_tools import SearchTools


# 配置日志
logger = logging.getLogger(__name__)


# ==================== Pydantic Models for Tool Inputs ====================


class GetLatestNewsInput(BaseModel):
    """获取最新新闻的输入参数"""

    platforms: Optional[List[str]] = Field(
        None,
        description="平台ID列表，如 ['zhihu', 'weibo', 'douyin']。不指定时使用所有平台。",
    )
    limit: int = Field(
        50, ge=1, le=1000, description="返回条数限制，默认50，最大1000"
    )
    include_url: bool = Field(False, description="是否包含URL链接，默认False")


class AnalyzeTrendInput(BaseModel):
    """分析趋势的输入参数"""

    topic: str = Field(..., description="话题关键词，必须提供")
    analysis_type: str = Field(
        "trend",
        description="分析类型: trend(趋势), lifecycle(生命周期), viral(异常热度), predict(预测)",
    )
    date_range: Optional[Dict[str, str]] = Field(
        None, description='日期范围，格式: {"start": "2025-01-01", "end": "2025-01-07"}'
    )


class SearchNewsInput(BaseModel):
    """搜索新闻的输入参数"""

    keyword: str = Field(..., description="搜索关键词")
    platforms: Optional[List[str]] = Field(None, description="平台ID列表")
    limit: int = Field(50, ge=1, le=1000, description="返回条数限制")


# ==================== LangChain Tools ====================


class GetLatestNewsTool(BaseTool):
    """获取最新新闻的 LangChain Tool"""

    name: str = "get_latest_news"
    description: str = """
    获取 TrendRadar 爬取的最新新闻数据。

    使用场景：
    - 用户询问"今天有什么热点"
    - 需要了解当前各平台的热门新闻
    - 分析最新的新闻趋势

    参数说明：
    - platforms: 可选，平台列表如 ['zhihu', 'weibo']
    - limit: 返回数量，默认50
    - include_url: 是否包含链接，默认False
    """
    args_schema: Type[BaseModel] = GetLatestNewsInput
    data_tools: Optional[DataQueryTools] = None

    def __init__(self, project_root: Optional[str] = None):
        super().__init__()
        self.data_tools = DataQueryTools(project_root)

    def _run(
        self,
        platforms: Optional[List[str]] = None,
        limit: int = 50,
        include_url: bool = False,
    ) -> str:
        """同步执行"""
        try:
            logger.info(
                f"Fetching latest news: platforms={platforms}, limit={limit}, include_url={include_url}"
            )
            result = self.data_tools.get_latest_news(
                platforms=platforms, limit=limit, include_url=include_url
            )

            # 记录成功结果
            news_count = len(result.get("news", [])) if isinstance(result, dict) else 0
            logger.info(f"Successfully fetched {news_count} news items")

            return json.dumps(result, ensure_ascii=False, indent=2)

        except Exception as e:
            error_msg = f"Failed to fetch latest news: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return json.dumps(
                {
                    "error": error_msg,
                    "error_type": type(e).__name__,
                    "success": False,
                },
                ensure_ascii=False,
            )

    async def _arun(
        self,
        platforms: Optional[List[str]] = None,
        limit: int = 50,
        include_url: bool = False,
    ) -> str:
        """真正的异步执行"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._run, platforms, limit, include_url
            )
            return result
        except Exception as e:
            error_msg = f"Async execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return json.dumps(
                {"error": error_msg, "success": False}, ensure_ascii=False
            )


class AnalyzeTrendTool(BaseTool):
    """分析话题趋势的 LangChain Tool"""

    name: str = "analyze_topic_trend"
    description: str = """
    分析特定话题的趋势变化。

    使用场景：
    - 用户询问"人工智能最近的热度如何"
    - 追踪某个话题的生命周期
    - 检测突然爆火的话题
    - 预测未来可能的热点

    参数说明：
    - topic: 话题关键词（必需）
    - analysis_type: 分析类型，可选值：
      * trend: 热度趋势分析
      * lifecycle: 生命周期分析
      * viral: 异常热度检测
      * predict: 话题预测
    - date_range: 日期范围（可选）
    """
    args_schema: Type[BaseModel] = AnalyzeTrendInput
    analytics_tools: Optional[AnalyticsTools] = None

    def __init__(self, project_root: Optional[str] = None):
        super().__init__()
        self.analytics_tools = AnalyticsTools(project_root)

    def _run(
        self,
        topic: str,
        analysis_type: str = "trend",
        date_range: Optional[Dict[str, str]] = None,
    ) -> str:
        """同步执行"""
        try:
            logger.info(
                f"Analyzing trend: topic='{topic}', type={analysis_type}, date_range={date_range}"
            )
            result = self.analytics_tools.analyze_topic_trend_unified(
                topic=topic, analysis_type=analysis_type, date_range=date_range
            )

            logger.info(f"Successfully analyzed trend for topic '{topic}'")
            return json.dumps(result, ensure_ascii=False, indent=2)

        except Exception as e:
            error_msg = f"Failed to analyze trend for '{topic}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return json.dumps(
                {
                    "error": error_msg,
                    "error_type": type(e).__name__,
                    "success": False,
                    "topic": topic,
                },
                ensure_ascii=False,
            )

    async def _arun(
        self,
        topic: str,
        analysis_type: str = "trend",
        date_range: Optional[Dict[str, str]] = None,
    ) -> str:
        """真正的异步执行"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._run, topic, analysis_type, date_range
            )
            return result
        except Exception as e:
            error_msg = f"Async execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return json.dumps(
                {"error": error_msg, "success": False}, ensure_ascii=False
            )


class SearchNewsTool(BaseTool):
    """搜索新闻的 LangChain Tool"""

    name: str = "search_news"
    description: str = """
    在已爬取的新闻中搜索特定关键词。

    使用场景：
    - 用户询问"搜索关于比特币的新闻"
    - 查找包含特定关键词的新闻
    - 跨平台搜索相关内容

    参数说明：
    - keyword: 搜索关键词（必需）
    - platforms: 平台列表（可选）
    - limit: 返回数量，默认50
    """
    args_schema: Type[BaseModel] = SearchNewsInput
    search_tools: Optional[SearchTools] = None

    def __init__(self, project_root: Optional[str] = None):
        super().__init__()
        self.search_tools = SearchTools(project_root)

    def _run(
        self,
        keyword: str,
        platforms: Optional[List[str]] = None,
        limit: int = 50,
    ) -> str:
        """同步执行"""
        try:
            logger.info(
                f"Searching news: keyword='{keyword}', platforms={platforms}, limit={limit}"
            )
            result = self.search_tools.search_news_by_keyword(
                keyword=keyword, platforms=platforms, limit=limit
            )

            # 记录成功结果
            news_count = len(result.get("news", [])) if isinstance(result, dict) else 0
            logger.info(f"Found {news_count} news items for keyword '{keyword}'")

            return json.dumps(result, ensure_ascii=False, indent=2)

        except Exception as e:
            error_msg = f"Failed to search news for '{keyword}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return json.dumps(
                {
                    "error": error_msg,
                    "error_type": type(e).__name__,
                    "success": False,
                    "keyword": keyword,
                },
                ensure_ascii=False,
            )

    async def _arun(
        self,
        keyword: str,
        platforms: Optional[List[str]] = None,
        limit: int = 50,
    ) -> str:
        """真正的异步执行"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._run, keyword, platforms, limit
            )
            return result
        except Exception as e:
            error_msg = f"Async execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return json.dumps(
                {"error": error_msg, "success": False}, ensure_ascii=False
            )


# ==================== 工具集合（带缓存） ====================

_tools_cache: Dict[Optional[str], List[BaseTool]] = {}


def get_all_trendradar_tools(project_root: Optional[str] = None) -> List[BaseTool]:
    """
    获取所有 TrendRadar 工具（带缓存优化）

    Args:
        project_root: 项目根目录

    Returns:
        工具列表
    """
    cache_key = project_root or "default"

    if cache_key not in _tools_cache:
        logger.info(f"Creating new tool instances for project_root={project_root}")
        _tools_cache[cache_key] = [
            GetLatestNewsTool(project_root),
            AnalyzeTrendTool(project_root),
            SearchNewsTool(project_root),
        ]
    else:
        logger.debug(f"Using cached tools for project_root={project_root}")

    return _tools_cache[cache_key]
