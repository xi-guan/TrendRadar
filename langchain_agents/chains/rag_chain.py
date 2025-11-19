"""
RAG Chain (Retrieval-Augmented Generation)

使用向量检索增强生成，提供基于新闻数据库的智能问答。
"""

import logging
from typing import List, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document

from langchain_agents.config import get_config
from langchain_agents.vectorstore.chroma_store import NewsVectorStore


logger = logging.getLogger(__name__)


# ==================== Prompt Templates ====================


RAG_SYSTEM_PROMPT = """你是一位专业的新闻分析助手，能够基于新闻数据库回答用户问题。

你的任务：
1. 仔细阅读提供的新闻上下文
2. 基于上下文回答用户问题
3. 如果上下文中没有足够信息，诚实告知
4. 回答要准确、简洁、有据可查

重要规则：
- 只基于提供的上下文回答，不要编造信息
- 如果信息不足，说明需要更多信息
- 可以引用具体的新闻来源和时间
- 保持客观中立的态度
"""


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human", """相关新闻上下文：
{context}

用户问题：{question}

请基于以上新闻上下文回答用户问题："""),
])


# ==================== Helper Functions ====================


def format_docs(docs: List[Document]) -> str:
    """
    格式化文档为上下文字符串

    Args:
        docs: 文档列表

    Returns:
        格式化的上下文字符串
    """
    if not docs:
        return "（未找到相关新闻）"

    formatted = []
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("title", "N/A")
        source = doc.metadata.get("source", "N/A")
        timestamp = doc.metadata.get("timestamp", "N/A")
        content = doc.page_content[:500]  # 限制长度

        formatted.append(
            f"[{i}] 标题: {title}\n"
            f"    来源: {source}\n"
            f"    时间: {timestamp}\n"
            f"    内容: {content}..."
        )

    return "\n\n".join(formatted)


# ==================== RAG Chain ====================


class RAGChain:
    """
    RAG Chain (Retrieval-Augmented Generation)

    功能：
    - 向量检索：从新闻数据库检索相关文档
    - 上下文增强：将检索到的文档作为上下文
    - 智能生成：基于上下文生成准确答案
    """

    def __init__(
        self,
        vectorstore: NewsVectorStore,
        llm=None,
        k: int = 5,
    ):
        """
        初始化 RAG Chain

        Args:
            vectorstore: 向量存储实例
            llm: LLM 实例（可选）
            k: 检索的文档数量
        """
        self.vectorstore = vectorstore
        self.k = k

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

        # 构建 LCEL Chain
        self._build_chain()

        logger.info(f"RAGChain initialized with k={k}")

    def _build_chain(self):
        """构建 LCEL Chain"""
        # 获取 retriever
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.k}
        )

        # 构建 RAG chain (LangChain 1.0 LCEL 最佳实践)
        self.chain = (
            RunnableParallel({
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            })
            | RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )

        logger.info("LCEL RAG chain built successfully")

    def ask(self, question: str) -> str:
        """
        提问

        Args:
            question: 用户问题

        Returns:
            AI 回答

        Raises:
            ValueError: 如果问题为空
        """
        # 输入验证 (从 P0 修复学到的经验)
        if not question or not question.strip():
            raise ValueError("question cannot be empty")

        try:
            logger.info(f"Processing RAG query: {question}")

            result = self.chain.invoke(question)

            logger.info("RAG query processed successfully")

            return result

        except Exception as e:
            error_msg = f"Failed to process RAG query: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise

    async def aask(self, question: str) -> str:
        """
        异步提问

        Args:
            question: 用户问题

        Returns:
            AI 回答

        Raises:
            ValueError: 如果问题为空
        """
        # 输入验证
        if not question or not question.strip():
            raise ValueError("question cannot be empty")

        try:
            logger.info(f"Processing async RAG query: {question}")

            result = await self.chain.ainvoke(question)

            logger.info("Async RAG query processed successfully")

            return result

        except Exception as e:
            error_msg = f"Failed to process async RAG query: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise

    def ask_with_sources(self, question: str) -> Dict[str, Any]:
        """
        提问并返回来源文档

        Args:
            question: 用户问题

        Returns:
            包含答案和来源的字典
        """
        # 输入验证
        if not question or not question.strip():
            raise ValueError("question cannot be empty")

        try:
            logger.info(f"Processing RAG query with sources: {question}")

            # 检索文档
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.k}
            )
            docs = retriever.invoke(question)

            # 生成答案
            context = format_docs(docs)
            answer = self.llm.invoke([
                ("system", RAG_SYSTEM_PROMPT),
                ("human", f"相关新闻上下文：\n{context}\n\n用户问题：{question}\n\n请基于以上新闻上下文回答用户问题："),
            ]).content

            logger.info("RAG query with sources processed successfully")

            return {
                "answer": answer,
                "sources": [
                    {
                        "title": doc.metadata.get("title", "N/A"),
                        "source": doc.metadata.get("source", "N/A"),
                        "timestamp": doc.metadata.get("timestamp", "N/A"),
                        "url": doc.metadata.get("url", ""),
                        "content": doc.page_content[:200],
                    }
                    for doc in docs
                ],
                "num_sources": len(docs),
            }

        except Exception as e:
            error_msg = f"Failed to process RAG query with sources: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise


class ConversationalRAGChain:
    """
    对话式 RAG Chain

    在 RAG 基础上增加对话历史管理。
    """

    def __init__(
        self,
        vectorstore: NewsVectorStore,
        llm=None,
        k: int = 5,
        max_history: int = 10,
    ):
        """
        初始化对话式 RAG Chain

        Args:
            vectorstore: 向量存储实例
            llm: LLM 实例（可选）
            k: 检索的文档数量
            max_history: 最大对话历史长度
        """
        # 初始化基础 RAG chain
        self.rag_chain = RAGChain(vectorstore=vectorstore, llm=llm, k=k)

        # 初始化对话记忆
        from langchain_agents.memory.conversation_buffer import ConversationBufferWindowMemory
        self.memory = ConversationBufferWindowMemory(k=max_history)

        logger.info(f"ConversationalRAGChain initialized with max_history={max_history}")

    def ask(self, question: str) -> str:
        """
        提问（带对话历史）

        Args:
            question: 用户问题

        Returns:
            AI 回答
        """
        try:
            # 获取对话历史
            history = self.memory.get_formatted_messages()

            # 如果有历史，将历史添加到问题前
            if history:
                enhanced_question = f"对话历史：\n{history}\n\n当前问题：{question}"
            else:
                enhanced_question = question

            # 调用 RAG chain
            answer = self.rag_chain.ask(enhanced_question)

            # 更新记忆
            self.memory.add_user_message(question)
            self.memory.add_ai_message(answer)

            return answer

        except Exception as e:
            error_msg = f"Failed to process conversational RAG query: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise

    async def aask(self, question: str) -> str:
        """异步提问（带对话历史）"""
        try:
            history = self.memory.get_formatted_messages()

            if history:
                enhanced_question = f"对话历史：\n{history}\n\n当前问题：{question}"
            else:
                enhanced_question = question

            answer = await self.rag_chain.aask(enhanced_question)

            self.memory.add_user_message(question)
            self.memory.add_ai_message(answer)

            return answer

        except Exception as e:
            error_msg = f"Failed to process async conversational RAG query: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise

    def clear_history(self):
        """清空对话历史"""
        self.memory.clear()
        logger.info("Conversation history cleared")

    def get_history(self):
        """获取对话历史"""
        return self.memory.messages


# ==================== 便捷函数 ====================


def create_rag_chain(
    vectorstore: NewsVectorStore,
    llm=None,
    k: int = 5,
    conversational: bool = False,
    max_history: int = 10,
):
    """
    创建 RAG Chain 的工厂函数

    Args:
        vectorstore: 向量存储实例
        llm: LLM 实例（可选）
        k: 检索的文档数量
        conversational: 是否使用对话式 RAG
        max_history: 最大对话历史长度（仅用于对话式）

    Returns:
        RAGChain 或 ConversationalRAGChain 实例
    """
    if conversational:
        logger.info("Creating ConversationalRAGChain")
        return ConversationalRAGChain(
            vectorstore=vectorstore,
            llm=llm,
            k=k,
            max_history=max_history,
        )
    else:
        logger.info("Creating RAGChain")
        return RAGChain(
            vectorstore=vectorstore,
            llm=llm,
            k=k,
        )
