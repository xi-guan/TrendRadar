"""
新闻分析 Agent

使用 LangChain 1.0 的 create_agent API，
提供智能新闻分析和摘要功能。
"""

from typing import Optional, List, Dict, Any
from langchain_core.tools import BaseTool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from .base_agent import BaseAgent
from ..tools.trendradar_tools import get_all_trendradar_tools


class NewsAnalystAgent(BaseAgent):
    """新闻分析 Agent"""

    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None,
        project_root: Optional[str] = None,
    ):
        """
        初始化新闻分析 Agent

        Args:
            tools: 自定义工具列表，不提供则使用默认的 TrendRadar 工具
            project_root: TrendRadar 项目根目录
        """
        # 使用默认工具
        if tools is None:
            tools = get_all_trendradar_tools(project_root)

        # 系统提示词
        system_prompt = """你是一位专业的新闻分析师。你的任务是：

1. **准确分析**：客观分析新闻热点，不带偏见
2. **提取关键信息**：识别新闻中的核心要点
3. **生成摘要**：用简洁的语言总结新闻内容
4. **趋势洞察**：发现热点背后的趋势和模式

可用工具：
- get_latest_news: 获取最新新闻
- analyze_topic_trend: 分析话题趋势
- search_news: 搜索相关新闻

回答要求：
- 保持客观中立
- 使用简洁明了的语言
- 提供数据支撑的结论
- 中文回答
"""

        super().__init__(tools=tools, system_prompt=system_prompt)

        # 使用 LangChain 1.0 的 create_agent
        self.agent = create_agent(
            model=self.llm, tools=self.tools, system_prompt=self.system_prompt
        )

    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        调用 Agent

        Args:
            input_data: 输入数据，格式: {"input": "用户问题"}

        Returns:
            Agent 的输出结果
        """
        return self.agent.invoke(input_data)

    def analyze_daily_news(self, date: Optional[str] = None) -> str:
        """
        分析每日新闻摘要

        Args:
            date: 日期，格式 "YYYY-MM-DD"，不提供则分析今天

        Returns:
            新闻摘要文本
        """
        date_str = date or "今天"
        prompt = f"请分析{date_str}的新闻热点，生成一份简洁的每日摘要。包括：1）主要热点话题 2）重要新闻概述 3）趋势观察"

        result = self.invoke({"input": prompt})
        return result.get("output", "")

    def compare_platforms(self, topic: str) -> str:
        """
        对比不同平台对某话题的关注度

        Args:
            topic: 话题关键词

        Returns:
            对比分析结果
        """
        prompt = f'请分析话题"{topic}"在不同平台的热度和讨论情况，对比各平台的关注点差异。'

        result = self.invoke({"input": prompt})
        return result.get("output", "")

    def predict_trend(self, topic: str) -> str:
        """
        预测话题趋势

        Args:
            topic: 话题关键词

        Returns:
            趋势预测结果
        """
        prompt = f'基于历史数据，分析话题"{topic}"的未来趋势，预测其热度变化。'

        result = self.invoke({"input": prompt})
        return result.get("output", "")
