"""
新闻摘要 Chain

使用 LangChain 1.0 LCEL (LangChain Expression Language) 实现智能摘要功能。
"""

import logging
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_agents.config import get_config


logger = logging.getLogger(__name__)


# ==================== Prompt Templates ====================

SINGLE_NEWS_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一位专业的新闻编辑，擅长提炼新闻要点。

任务：将新闻内容概括为简洁的摘要（2-3句话）。

要求：
1. 提取核心信息（谁、什么、何时、何地、为何、如何）
2. 保持客观中立，不添加个人观点
3. 使用清晰简洁的语言
4. 控制在 100 字以内"""),
    ("human", "新闻标题：{title}\n\n新闻内容：{content}\n\n请提供摘要："),
])


MULTIPLE_NEWS_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一位专业的新闻分析师，擅长整合多条新闻。

任务：将多条相关新闻整合为一份综合摘要。

要求：
1. 识别共同主题和关键趋势
2. 提取最重要的信息点
3. 按时间或重要性组织内容
4. 总长度控制在 300 字以内"""),
    ("human", """以下是 {count} 条关于 "{topic}" 的新闻：

{news_list}

请提供综合摘要："""),
])


TREND_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一位资深的趋势分析专家。

任务：基于新闻数据分析话题的发展趋势。

要求：
1. 识别话题的热度变化
2. 分析讨论重点的演变
3. 预测可能的发展方向
4. 提供数据支持的洞察"""),
    ("human", """话题：{topic}

趋势数据：
{trend_data}

相关新闻（最近 {news_count} 条）：
{news_summary}

请提供趋势分析摘要："""),
])


# ==================== Chain Implementations ====================


class NewsSummaryChain:
    """新闻摘要 Chain（单条新闻）"""

    def __init__(self, llm=None):
        """
        初始化摘要链

        Args:
            llm: LLM 实例，如果为 None 则使用默认配置
        """
        if llm is None:
            config = get_config()
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=config.llm.model,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
                timeout=config.llm.timeout,
            )

        self.llm = llm

        # 使用 LCEL 构建 Chain
        self.chain = (
            SINGLE_NEWS_SUMMARY_PROMPT
            | self.llm
            | StrOutputParser()
        )

        logger.info("NewsSummaryChain initialized")

    def summarize(self, title: str, content: str) -> str:
        """
        生成单条新闻摘要

        Args:
            title: 新闻标题
            content: 新闻内容

        Returns:
            摘要文本

        Raises:
            ValueError: 如果输入无效

        Note:
            P0 修复: 添加输入验证避免浪费 API 成本
        """
        # P0 修复: 输入验证
        if not title or not title.strip():
            raise ValueError("title cannot be empty")
        if not content or not content.strip():
            raise ValueError("content cannot be empty")
        if len(content.strip()) < 10:
            raise ValueError("content too short to summarize (minimum 10 characters)")

        try:
            logger.info(f"Summarizing news: {title[:50]}...")

            result = self.chain.invoke({
                "title": title,
                "content": content,
            })

            logger.info("Summary generated successfully")
            return result

        except Exception as e:
            error_msg = f"Failed to generate summary: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise

    async def asummarize(self, title: str, content: str) -> str:
        """
        异步生成摘要

        Args:
            title: 新闻标题
            content: 新闻内容

        Returns:
            摘要文本

        Raises:
            ValueError: 如果输入无效

        Note:
            P0 修复: 添加输入验证避免浪费 API 成本
        """
        # P0 修复: 输入验证
        if not title or not title.strip():
            raise ValueError("title cannot be empty")
        if not content or not content.strip():
            raise ValueError("content cannot be empty")
        if len(content.strip()) < 10:
            raise ValueError("content too short to summarize (minimum 10 characters)")

        try:
            logger.info(f"Async summarizing news: {title[:50]}...")

            result = await self.chain.ainvoke({
                "title": title,
                "content": content,
            })

            logger.info("Async summary generated successfully")
            return result

        except Exception as e:
            error_msg = f"Failed to generate async summary: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise


class MultipleNewsSummaryChain:
    """多新闻综合摘要 Chain"""

    def __init__(self, llm=None):
        """
        初始化综合摘要链

        Args:
            llm: LLM 实例，如果为 None 则使用默认配置
        """
        if llm is None:
            config = get_config()
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=config.llm.model,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens * 2,  # 综合摘要需要更多 tokens
                timeout=config.llm.timeout,
            )

        self.llm = llm

        # 使用 LCEL 构建 Chain
        self.chain = (
            MULTIPLE_NEWS_SUMMARY_PROMPT
            | self.llm
            | StrOutputParser()
        )

        logger.info("MultipleNewsSummaryChain initialized")

    def summarize(self, topic: str, news_list: List[Dict[str, Any]]) -> str:
        """
        生成多条新闻的综合摘要

        Args:
            topic: 主题名称
            news_list: 新闻列表，每条新闻包含 title, content, timestamp 等字段

        Returns:
            综合摘要文本
        """
        try:
            logger.info(f"Summarizing {len(news_list)} news items for topic: {topic}")

            # 格式化新闻列表
            formatted_news = "\n\n".join([
                f"{i+1}. 【{news.get('title', 'N/A')}】\n"
                f"   时间: {news.get('timestamp', 'N/A')}\n"
                f"   内容: {news.get('content', 'N/A')[:200]}..."
                for i, news in enumerate(news_list[:10])  # 限制最多 10 条
            ])

            result = self.chain.invoke({
                "topic": topic,
                "count": len(news_list),
                "news_list": formatted_news,
            })

            logger.info("Comprehensive summary generated successfully")
            return result

        except Exception as e:
            error_msg = f"Failed to generate comprehensive summary: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise

    async def asummarize(self, topic: str, news_list: List[Dict[str, Any]]) -> str:
        """异步生成综合摘要"""
        try:
            logger.info(f"Async summarizing {len(news_list)} news items for topic: {topic}")

            formatted_news = "\n\n".join([
                f"{i+1}. 【{news.get('title', 'N/A')}】\n"
                f"   时间: {news.get('timestamp', 'N/A')}\n"
                f"   内容: {news.get('content', 'N/A')[:200]}..."
                for i, news in enumerate(news_list[:10])
            ])

            result = await self.chain.ainvoke({
                "topic": topic,
                "count": len(news_list),
                "news_list": formatted_news,
            })

            logger.info("Async comprehensive summary generated successfully")
            return result

        except Exception as e:
            error_msg = f"Failed to generate async comprehensive summary: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise


class TrendAnalysisSummaryChain:
    """趋势分析摘要 Chain"""

    def __init__(self, llm=None):
        """
        初始化趋势分析链

        Args:
            llm: LLM 实例，如果为 None 则使用默认配置
        """
        if llm is None:
            config = get_config()
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=config.llm.model,
                temperature=config.llm.temperature + 0.1,  # 趋势分析需要稍高创造性
                max_tokens=config.llm.max_tokens * 2,
                timeout=config.llm.timeout,
            )

        self.llm = llm

        # 使用 LCEL 构建 Chain
        self.chain = (
            TREND_SUMMARY_PROMPT
            | self.llm
            | StrOutputParser()
        )

        logger.info("TrendAnalysisSummaryChain initialized")

    def analyze(
        self,
        topic: str,
        trend_data: Dict[str, Any],
        news_list: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        生成趋势分析摘要

        Args:
            topic: 话题名称
            trend_data: 趋势数据（包含热度、时间序列等）
            news_list: 相关新闻列表（可选）

        Returns:
            趋势分析摘要
        """
        try:
            logger.info(f"Analyzing trend for topic: {topic}")

            # 格式化趋势数据
            trend_text = f"""
- 热度值: {trend_data.get('heat', 'N/A')}
- 新闻数量: {trend_data.get('news_count', 'N/A')}
- 时间跨度: {trend_data.get('date_range', {}).get('start', 'N/A')} 至 {trend_data.get('date_range', {}).get('end', 'N/A')}
- 趋势变化: {trend_data.get('trend_direction', 'N/A')}
            """.strip()

            # 格式化新闻摘要
            news_summary = "无相关新闻"
            if news_list:
                news_summary = "\n".join([
                    f"- {news.get('title', 'N/A')} ({news.get('timestamp', 'N/A')})"
                    for news in news_list[:5]  # 只显示最近 5 条
                ])

            result = self.chain.invoke({
                "topic": topic,
                "trend_data": trend_text,
                "news_count": len(news_list) if news_list else 0,
                "news_summary": news_summary,
            })

            logger.info("Trend analysis summary generated successfully")
            return result

        except Exception as e:
            error_msg = f"Failed to generate trend analysis summary: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise

    async def aanalyze(
        self,
        topic: str,
        trend_data: Dict[str, Any],
        news_list: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """异步生成趋势分析摘要"""
        try:
            logger.info(f"Async analyzing trend for topic: {topic}")

            trend_text = f"""
- 热度值: {trend_data.get('heat', 'N/A')}
- 新闻数量: {trend_data.get('news_count', 'N/A')}
- 时间跨度: {trend_data.get('date_range', {}).get('start', 'N/A')} 至 {trend_data.get('date_range', {}).get('end', 'N/A')}
- 趋势变化: {trend_data.get('trend_direction', 'N/A')}
            """.strip()

            news_summary = "无相关新闻"
            if news_list:
                news_summary = "\n".join([
                    f"- {news.get('title', 'N/A')} ({news.get('timestamp', 'N/A')})"
                    for news in news_list[:5]
                ])

            result = await self.chain.ainvoke({
                "topic": topic,
                "trend_data": trend_text,
                "news_count": len(news_list) if news_list else 0,
                "news_summary": news_summary,
            })

            logger.info("Async trend analysis summary generated successfully")
            return result

        except Exception as e:
            error_msg = f"Failed to generate async trend analysis summary: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise


# ==================== 便捷函数 ====================


def create_summary_chain(chain_type: str = "single", llm=None):
    """
    创建摘要链的工厂函数

    Args:
        chain_type: 链类型 ("single", "multiple", "trend")
        llm: 自定义 LLM 实例（可选）

    Returns:
        对应的 Chain 实例
    """
    chain_map = {
        "single": NewsSummaryChain,
        "multiple": MultipleNewsSummaryChain,
        "trend": TrendAnalysisSummaryChain,
    }

    if chain_type not in chain_map:
        raise ValueError(f"Invalid chain_type: {chain_type}. Must be one of {list(chain_map.keys())}")

    logger.info(f"Creating {chain_type} summary chain")
    return chain_map[chain_type](llm=llm)
