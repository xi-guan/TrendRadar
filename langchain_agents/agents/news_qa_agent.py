"""
News QA Agent

对话式新闻问答 Agent，使用 LangGraph 实现多轮对话和工具调用。
"""

import logging
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from operator import add

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from langchain_agents.config import get_config
from langchain_agents.memory.conversation_buffer import ConversationBufferMemory
from langchain_agents.tools.trendradar_tools import get_all_trendradar_tools


logger = logging.getLogger(__name__)


# ==================== State Definition ====================


class NewsQAState(TypedDict):
    """
    NewsQA Agent State

    使用 TypedDict + Annotated 实现状态管理 (LangGraph 1.0 最佳实践)
    """
    # 对话消息列表
    messages: Annotated[List[Any], add]

    # 用户查询
    query: str

    # 工具调用结果
    tool_results: Optional[List[Dict[str, Any]]]

    # 最终答案
    answer: Optional[str]

    # 错误信息
    errors: Annotated[List[str], add]

    # 完成的步骤
    steps_completed: Annotated[List[str], add]


# ==================== Prompt Templates ====================


QA_SYSTEM_PROMPT = """你是一位专业的新闻问答助手，能够帮助用户查询和理解新闻信息。

你的能力：
1. 搜索新闻：使用 search_news 工具搜索相关新闻
2. 分析话题：使用 get_topic_info 工具获取话题详情
3. 趋势分析：使用 analyze_trend 工具分析趋势
4. 对话理解：理解多轮对话上下文

回答要求：
- 基于事实，引用具体新闻来源
- 简洁明了，直接回答用户问题
- 如果信息不足，诚实告知并建议如何获取更多信息
- 保持专业和客观的态度

你可以使用以下工具：
{tools}

当前时间：{current_time}
"""


ANSWER_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}"),
])


# ==================== Agent Nodes ====================


def should_use_tools(state: NewsQAState) -> str:
    """
    决定是否需要使用工具

    Args:
        state: 当前状态

    Returns:
        下一个节点名称 ("use_tools" 或 "generate_answer")
    """
    query = state["query"].lower()

    # 如果查询包含搜索、分析等关键词，使用工具
    tool_keywords = [
        "搜索", "查找", "找", "search",
        "什么新闻", "有哪些", "告诉我",
        "趋势", "热度", "分析",
        "话题", "事件"
    ]

    for keyword in tool_keywords:
        if keyword in query:
            logger.info(f"Query requires tools: '{query}'")
            return "use_tools"

    # 否则直接生成答案（基于对话历史）
    logger.info(f"Query doesn't require tools: '{query}'")
    return "generate_answer"


def use_tools_node(state: NewsQAState) -> NewsQAState:
    """
    使用工具搜索信息

    Args:
        state: 当前状态

    Returns:
        更新后的状态
    """
    try:
        logger.info("Using tools to search for information")

        query = state["query"]
        tools = get_all_trendradar_tools()

        # 简化：默认使用 search_news 工具
        search_tool = next((t for t in tools if t.name == "search_news"), None)

        if search_tool is None:
            error_msg = "search_news tool not found"
            logger.error(error_msg)
            return {
                **state,
                "tool_results": [],
                "errors": [error_msg],
                "steps_completed": ["use_tools"],
            }

        # 执行搜索
        result_json = search_tool._run(keyword=query, platforms=None, limit=5)

        logger.info(f"Tool search completed: {len(result_json.get('data', []))} results")

        return {
            **state,
            "tool_results": result_json.get("data", []),
            "steps_completed": ["use_tools"],
        }

    except Exception as e:
        error_msg = f"Tool execution failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            **state,
            "tool_results": [],
            "errors": [error_msg],
            "steps_completed": ["use_tools"],
        }


def generate_answer_node(state: NewsQAState) -> NewsQAState:
    """
    生成最终答案

    Args:
        state: 当前状态

    Returns:
        更新后的状态
    """
    try:
        logger.info("Generating answer")

        # 获取配置和 LLM
        config = get_config()
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            timeout=config.llm.timeout,
        )

        # 构建上下文
        query = state["query"]
        tool_results = state.get("tool_results", [])
        messages = state.get("messages", [])

        # 构建工具结果摘要
        context = ""
        if tool_results:
            context = "相关新闻：\n"
            for i, news in enumerate(tool_results[:5], 1):
                title = news.get("title", "N/A")
                content = news.get("content", "N/A")
                source = news.get("source", "N/A")
                context += f"\n{i}. {title}\n   来源: {source}\n   内容: {content[:150]}...\n"

        # 构建 prompt
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        tools_description = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in get_all_trendradar_tools()
        ])

        # 使用 LCEL 构建 chain
        if context:
            # 如果有工具结果，添加到消息中
            full_query = f"{query}\n\n{context}"
        else:
            full_query = query

        # 简化：直接调用 LLM
        response = llm.invoke([
            SystemMessage(content=QA_SYSTEM_PROMPT.format(
                tools=tools_description,
                current_time=current_time
            )),
            *messages,
            HumanMessage(content=full_query),
        ])

        answer = response.content

        logger.info(f"Answer generated: {len(answer)} characters")

        return {
            **state,
            "answer": answer,
            "messages": [HumanMessage(content=query), AIMessage(content=answer)],
            "steps_completed": ["generate_answer"],
        }

    except Exception as e:
        error_msg = f"Answer generation failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            **state,
            "answer": f"抱歉，我在生成答案时遇到了问题：{str(e)}",
            "errors": [error_msg],
            "steps_completed": ["generate_answer"],
        }


# ==================== Agent Class ====================


class NewsQAAgent:
    """
    新闻问答 Agent

    功能：
    - 对话式问答
    - 工具调用（搜索新闻、获取话题、分析趋势）
    - 对话历史管理
    - LangGraph 工作流
    """

    def __init__(
        self,
        llm=None,
        memory: Optional[ConversationBufferMemory] = None,
        max_history: int = 10,
    ):
        """
        初始化 NewsQA Agent

        Args:
            llm: LLM 实例（可选）
            memory: 对话记忆实例（可选）
            max_history: 最大对话历史长度
        """
        # 初始化 LLM
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

        # 初始化记忆
        if memory is None:
            from langchain_agents.memory.conversation_buffer import ConversationBufferWindowMemory
            memory = ConversationBufferWindowMemory(k=max_history)

        self.memory = memory
        self.max_history = max_history

        # 构建 LangGraph 工作流
        self._build_graph()

        logger.info("NewsQAAgent initialized")

    def _build_graph(self):
        """构建 LangGraph 工作流"""
        workflow = StateGraph(NewsQAState)

        # 添加节点
        workflow.add_node("use_tools", use_tools_node)
        workflow.add_node("generate_answer", generate_answer_node)

        # 设置入口点（条件路由）
        workflow.set_conditional_entry_point(
            should_use_tools,
            {
                "use_tools": "use_tools",
                "generate_answer": "generate_answer",
            }
        )

        # 添加边
        workflow.add_edge("use_tools", "generate_answer")
        workflow.add_edge("generate_answer", END)

        # 编译
        self.app = workflow.compile()

        logger.info("LangGraph workflow built successfully")

    def ask(self, query: str) -> str:
        """
        提问

        Args:
            query: 用户问题

        Returns:
            AI 回答
        """
        try:
            logger.info(f"Processing query: {query}")

            # 获取对话历史
            chat_history = self.memory.messages

            # 初始化状态
            initial_state: NewsQAState = {
                "messages": chat_history,
                "query": query,
                "tool_results": None,
                "answer": None,
                "errors": [],
                "steps_completed": [],
            }

            # 运行工作流
            final_state = self.app.invoke(initial_state)

            answer = final_state.get("answer", "抱歉，我无法回答这个问题。")

            # 更新记忆
            self.memory.add_user_message(query)
            self.memory.add_ai_message(answer)

            logger.info("Query processed successfully")

            return answer

        except Exception as e:
            error_msg = f"Failed to process query: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"抱歉，处理您的问题时出错了：{str(e)}"

    async def aask(self, query: str) -> str:
        """
        异步提问

        Args:
            query: 用户问题

        Returns:
            AI 回答
        """
        try:
            logger.info(f"Processing async query: {query}")

            chat_history = self.memory.messages

            initial_state: NewsQAState = {
                "messages": chat_history,
                "query": query,
                "tool_results": None,
                "answer": None,
                "errors": [],
                "steps_completed": [],
            }

            final_state = await self.app.ainvoke(initial_state)

            answer = final_state.get("answer", "抱歉，我无法回答这个问题。")

            self.memory.add_user_message(query)
            self.memory.add_ai_message(answer)

            logger.info("Async query processed successfully")

            return answer

        except Exception as e:
            error_msg = f"Failed to process async query: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"抱歉，处理您的问题时出错了：{str(e)}"

    def clear_history(self):
        """清空对话历史"""
        self.memory.clear()
        logger.info("Conversation history cleared")

    def get_history(self) -> List[Any]:
        """
        获取对话历史

        Returns:
            消息列表
        """
        return self.memory.messages

    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            统计信息字典
        """
        memory_stats = self.memory.get_stats()

        return {
            "agent_type": "NewsQAAgent",
            "memory": memory_stats,
            "max_history": self.max_history,
        }


# ==================== 便捷函数 ====================


def create_news_qa_agent(
    llm=None,
    max_history: int = 10,
) -> NewsQAAgent:
    """
    创建 NewsQA Agent 的工厂函数

    Args:
        llm: 自定义 LLM 实例（可选）
        max_history: 最大对话历史长度

    Returns:
        NewsQAAgent 实例
    """
    logger.info("Creating NewsQAAgent")
    return NewsQAAgent(llm=llm, max_history=max_history)
