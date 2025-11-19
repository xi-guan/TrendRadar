"""
Trend Predictor Agent

使用 LangGraph 实现的趋势预测 Agent，具有多步推理和状态管理能力。
"""

import logging
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from operator import add

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_agents.config import get_config
from langchain_agents.tools.trendradar_tools import get_all_trendradar_tools


logger = logging.getLogger(__name__)


# ==================== State Definition ====================


class TrendPredictorState(TypedDict):
    """
    趋势预测 Agent 的状态

    使用 TypedDict 定义状态结构，Annotated 用于指定合并策略。
    """
    # 输入
    topic: str  # 要分析的话题
    analysis_depth: str  # 分析深度 ("quick", "standard", "deep")

    # 中间状态
    news_data: Optional[Dict[str, Any]]  # 收集的新闻数据
    trend_analysis: Optional[Dict[str, Any]]  # 趋势分析结果
    prediction: Optional[str]  # 预测结果
    confidence: Optional[float]  # 预测置信度

    # 输出
    final_report: Optional[str]  # 最终报告
    recommendations: Annotated[List[str], add]  # 建议列表（使用 add 进行合并）

    # 元数据
    steps_completed: Annotated[List[str], add]  # 已完成的步骤
    errors: Annotated[List[str], add]  # 错误记录


# ==================== Node Functions ====================


def collect_news_node(state: TrendPredictorState) -> TrendPredictorState:
    """
    节点1: 收集相关新闻

    使用 SearchNewsTool 收集话题相关的新闻数据。
    """
    topic = state["topic"]
    analysis_depth = state.get("analysis_depth", "standard")

    # 根据分析深度决定收集的新闻数量
    limit_map = {
        "quick": 20,
        "standard": 50,
        "deep": 100,
    }
    limit = limit_map.get(analysis_depth, 50)

    try:
        logger.info(f"Collecting news for topic: {topic} (limit={limit})")

        # 获取 TrendRadar 工具 (P0 修复: 添加默认值避免 StopIteration)
        tools = get_all_trendradar_tools()
        search_tool = next((t for t in tools if t.name == "search_news"), None)

        if search_tool is None:
            error_msg = "search_news tool not found in available tools"
            logger.error(error_msg)
            return {
                **state,
                "news_data": {"error": error_msg, "count": 0, "news": []},
                "errors": [error_msg],
                "steps_completed": ["collect_news"],
            }

        # 搜索新闻
        result_json = search_tool._run(keyword=topic, platforms=None, limit=limit)

        import json
        news_data = json.loads(result_json)

        logger.info(f"Collected {news_data.get('count', 0)} news items")

        return {
            **state,
            "news_data": news_data,
            "steps_completed": ["collect_news"],
        }

    except Exception as e:
        error_msg = f"Failed to collect news: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            **state,
            "news_data": {"error": error_msg, "count": 0, "news": []},
            "errors": [error_msg],
            "steps_completed": ["collect_news"],
        }


def analyze_trend_node(state: TrendPredictorState) -> TrendPredictorState:
    """
    节点2: 分析趋势

    使用 AnalyzeTrendTool 分析话题的趋势变化。
    """
    topic = state["topic"]
    news_data = state.get("news_data", {})

    try:
        logger.info(f"Analyzing trend for topic: {topic}")

        # 获取分析工具 (P0 修复: 添加默认值避免 StopIteration)
        tools = get_all_trendradar_tools()
        analyze_tool = next((t for t in tools if t.name == "analyze_topic_trend"), None)

        if analyze_tool is None:
            error_msg = "analyze_topic_trend tool not found in available tools"
            logger.error(error_msg)
            return {
                **state,
                "trend_analysis": {"error": error_msg},
                "errors": [error_msg],
                "steps_completed": ["analyze_trend"],
            }

        # 分析趋势
        result_json = analyze_tool._run(
            topic=topic,
            analysis_type="trend",
            date_range=None,
        )

        import json
        trend_analysis = json.loads(result_json)

        logger.info(f"Trend analysis completed for topic: {topic}")

        return {
            **state,
            "trend_analysis": trend_analysis,
            "steps_completed": ["analyze_trend"],
        }

    except Exception as e:
        error_msg = f"Failed to analyze trend: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            **state,
            "trend_analysis": {"error": error_msg},
            "errors": [error_msg],
            "steps_completed": ["analyze_trend"],
        }


def generate_prediction_node(state: TrendPredictorState) -> TrendPredictorState:
    """
    节点3: 生成预测

    使用 LLM 基于新闻数据和趋势分析生成预测。
    """
    topic = state["topic"]
    news_data = state.get("news_data", {})
    trend_analysis = state.get("trend_analysis", {})

    try:
        logger.info(f"Generating prediction for topic: {topic}")

        # 获取 LLM
        config = get_config()
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=config.llm.model,
            temperature=config.llm.temperature + 0.2,  # 预测需要更高创造性
            max_tokens=config.llm.max_tokens * 2,
        )

        # 构建 Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位资深的趋势分析专家，擅长基于数据预测未来发展。

任务：基于提供的新闻数据和趋势分析，预测话题的未来发展方向。

要求：
1. 分析当前趋势的驱动因素
2. 识别可能的转折点
3. 预测短期（1-2周）和中期（1-3个月）发展
4. 评估预测的置信度
5. 提供数据支持的理由"""),
            ("human", """话题：{topic}

新闻数据：
- 新闻数量：{news_count}
- 最新新闻：{latest_news}

趋势分析：
{trend_summary}

请提供详细的趋势预测："""),
        ])

        # 准备数据
        news_list = news_data.get("news", [])
        news_count = len(news_list)
        latest_news = "\n".join([
            f"- {news.get('title', 'N/A')}"
            for news in news_list[:5]
        ]) if news_list else "无新闻数据"

        trend_summary = f"""
- 热度：{trend_analysis.get('heat', 'N/A')}
- 趋势方向：{trend_analysis.get('trend_direction', 'N/A')}
- 新闻数量：{trend_analysis.get('news_count', 0)}
        """.strip()

        # 生成预测
        chain = prompt | llm | StrOutputParser()
        prediction = chain.invoke({
            "topic": topic,
            "news_count": news_count,
            "latest_news": latest_news,
            "trend_summary": trend_summary,
        })

        # 简单的置信度评估（基于新闻数量）
        if news_count >= 50:
            confidence = 0.85
        elif news_count >= 20:
            confidence = 0.70
        elif news_count >= 10:
            confidence = 0.55
        else:
            confidence = 0.40

        logger.info(f"Prediction generated with confidence {confidence:.2f}")

        return {
            **state,
            "prediction": prediction,
            "confidence": confidence,
            "steps_completed": ["generate_prediction"],
        }

    except Exception as e:
        error_msg = f"Failed to generate prediction: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            **state,
            "prediction": f"预测生成失败：{error_msg}",
            "confidence": 0.0,
            "errors": [error_msg],
            "steps_completed": ["generate_prediction"],
        }


def generate_recommendations_node(state: TrendPredictorState) -> TrendPredictorState:
    """
    节点4: 生成建议

    基于预测结果生成可操作的建议。
    """
    topic = state["topic"]
    prediction = state.get("prediction", "")
    confidence = state.get("confidence", 0.0)

    try:
        logger.info(f"Generating recommendations for topic: {topic}")

        # 基于置信度和预测内容生成建议
        recommendations = []

        if confidence >= 0.70:
            recommendations.append("✅ 预测置信度较高，建议密切关注该话题的发展")
        elif confidence >= 0.50:
            recommendations.append("⚠️  预测置信度中等，建议收集更多数据后再做判断")
        else:
            recommendations.append("❌ 预测置信度较低，建议谨慎参考")

        # 从预测中提取关键建议（简化版本）
        if "上升" in prediction or "增长" in prediction:
            recommendations.append("📈 趋势向上，建议提前布局相关领域")
        elif "下降" in prediction or "减少" in prediction:
            recommendations.append("📉 趋势向下，建议调整策略或规避风险")

        recommendations.append("🔍 建议定期（每周）更新趋势分析以跟踪变化")

        logger.info(f"Generated {len(recommendations)} recommendations")

        return {
            **state,
            "recommendations": recommendations,
            "steps_completed": ["generate_recommendations"],
        }

    except Exception as e:
        error_msg = f"Failed to generate recommendations: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            **state,
            "recommendations": [f"建议生成失败：{error_msg}"],
            "errors": [error_msg],
            "steps_completed": ["generate_recommendations"],
        }


def create_final_report_node(state: TrendPredictorState) -> TrendPredictorState:
    """
    节点5: 创建最终报告

    整合所有分析结果，生成结构化的最终报告。
    """
    topic = state["topic"]
    news_data = state.get("news_data", {})
    trend_analysis = state.get("trend_analysis", {})
    prediction = state.get("prediction", "无预测")
    confidence = state.get("confidence", 0.0)
    recommendations = state.get("recommendations", [])
    errors = state.get("errors", [])

    try:
        logger.info(f"Creating final report for topic: {topic}")

        # 构建报告
        report_sections = [
            "=" * 60,
            f"趋势预测报告：{topic}",
            "=" * 60,
            "",
            "## 1. 数据概况",
            f"- 新闻数量：{news_data.get('count', 0)}",
            f"- 话题热度：{trend_analysis.get('heat', 'N/A')}",
            f"- 预测置信度：{confidence:.2%}",
            "",
            "## 2. 趋势预测",
            prediction,
            "",
            "## 3. 建议",
        ]

        for rec in recommendations:
            report_sections.append(f"  {rec}")

        if errors:
            report_sections.extend([
                "",
                "## ⚠️  警告",
                "分析过程中出现以下错误：",
            ])
            for err in errors:
                report_sections.append(f"  - {err}")

        report_sections.extend([
            "",
            "=" * 60,
            f"已完成步骤：{', '.join(state.get('steps_completed', []))}",
            "=" * 60,
        ])

        final_report = "\n".join(report_sections)

        logger.info("Final report created successfully")

        return {
            **state,
            "final_report": final_report,
            "steps_completed": ["create_final_report"],
        }

    except Exception as e:
        error_msg = f"Failed to create final report: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            **state,
            "final_report": f"报告生成失败：{error_msg}",
            "errors": [error_msg],
            "steps_completed": ["create_final_report"],
        }


# ==================== Graph Construction ====================


def create_trend_predictor_graph():
    """
    创建趋势预测 LangGraph

    返回编译后的图
    """
    # 创建状态图
    workflow = StateGraph(TrendPredictorState)

    # 添加节点
    workflow.add_node("collect_news", collect_news_node)
    workflow.add_node("analyze_trend", analyze_trend_node)
    workflow.add_node("generate_prediction", generate_prediction_node)
    workflow.add_node("generate_recommendations", generate_recommendations_node)
    workflow.add_node("create_final_report", create_final_report_node)

    # 设置入口点
    workflow.set_entry_point("collect_news")

    # 添加边（定义执行流程）
    workflow.add_edge("collect_news", "analyze_trend")
    workflow.add_edge("analyze_trend", "generate_prediction")
    workflow.add_edge("generate_prediction", "generate_recommendations")
    workflow.add_edge("generate_recommendations", "create_final_report")
    workflow.add_edge("create_final_report", END)

    # 编译图
    app = workflow.compile()

    logger.info("TrendPredictorGraph compiled successfully")

    return app


# ==================== Agent Class ====================


class TrendPredictorAgent:
    """
    趋势预测 Agent

    使用 LangGraph 实现的有状态 Agent，可以执行多步推理。
    """

    def __init__(self):
        """初始化 Agent"""
        self.graph = create_trend_predictor_graph()
        logger.info("TrendPredictorAgent initialized")

    def predict(
        self,
        topic: str,
        analysis_depth: str = "standard",
    ) -> Dict[str, Any]:
        """
        执行趋势预测

        Args:
            topic: 要预测的话题
            analysis_depth: 分析深度 ("quick", "standard", "deep")

        Returns:
            预测结果字典
        """
        try:
            logger.info(f"Starting trend prediction for: {topic} (depth={analysis_depth})")

            # 初始化状态
            initial_state: TrendPredictorState = {
                "topic": topic,
                "analysis_depth": analysis_depth,
                "news_data": None,
                "trend_analysis": None,
                "prediction": None,
                "confidence": None,
                "final_report": None,
                "recommendations": [],
                "steps_completed": [],
                "errors": [],
            }

            # 执行图
            final_state = self.graph.invoke(initial_state)

            logger.info("Trend prediction completed successfully")

            return {
                "success": True,
                "topic": topic,
                "report": final_state.get("final_report", ""),
                "prediction": final_state.get("prediction", ""),
                "confidence": final_state.get("confidence", 0.0),
                "recommendations": final_state.get("recommendations", []),
                "steps_completed": final_state.get("steps_completed", []),
                "errors": final_state.get("errors", []),
            }

        except Exception as e:
            error_msg = f"Trend prediction failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "error": error_msg,
                "topic": topic,
            }

    async def apredict(
        self,
        topic: str,
        analysis_depth: str = "standard",
    ) -> Dict[str, Any]:
        """
        异步执行趋势预测

        Args:
            topic: 要预测的话题
            analysis_depth: 分析深度

        Returns:
            预测结果字典
        """
        try:
            logger.info(f"Starting async trend prediction for: {topic}")

            initial_state: TrendPredictorState = {
                "topic": topic,
                "analysis_depth": analysis_depth,
                "news_data": None,
                "trend_analysis": None,
                "prediction": None,
                "confidence": None,
                "final_report": None,
                "recommendations": [],
                "steps_completed": [],
                "errors": [],
            }

            # 异步执行图
            final_state = await self.graph.ainvoke(initial_state)

            logger.info("Async trend prediction completed successfully")

            return {
                "success": True,
                "topic": topic,
                "report": final_state.get("final_report", ""),
                "prediction": final_state.get("prediction", ""),
                "confidence": final_state.get("confidence", 0.0),
                "recommendations": final_state.get("recommendations", []),
                "steps_completed": final_state.get("steps_completed", []),
                "errors": final_state.get("errors", []),
            }

        except Exception as e:
            error_msg = f"Async trend prediction failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "error": error_msg,
                "topic": topic,
            }
