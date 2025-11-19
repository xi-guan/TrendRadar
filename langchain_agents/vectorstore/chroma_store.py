"""
Chroma Vector Store Integration

提供新闻文档的向量存储和检索功能。
"""

import logging
import threading
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_agents.config import get_config


logger = logging.getLogger(__name__)


class NewsVectorStore:
    """
    新闻向量存储

    功能：
    - 新闻文档的向量化存储
    - 基于语义的相似度检索
    - 持久化支持
    - 线程安全
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "trendradar_news",
        embeddings: Optional[Any] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        初始化向量存储

        Args:
            persist_directory: 持久化目录（None 表示内存模式）
            collection_name: 集合名称
            embeddings: 嵌入模型（None 则使用默认 OpenAI embeddings）
            chunk_size: 文档分块大小
            chunk_overlap: 分块重叠大小
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 初始化 embeddings
        if embeddings is None:
            config = get_config()
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",  # 最新的高性价比模型
                # OpenAI API key 从环境变量自动读取
            )

        self.embeddings = embeddings

        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
        )

        # 线程锁 (从 P0 修复学到的经验)
        self._lock = threading.RLock()

        # 初始化 Chroma
        self._initialize_vectorstore()

        logger.info(
            f"NewsVectorStore initialized: "
            f"collection={collection_name}, "
            f"persist_directory={persist_directory}, "
            f"chunk_size={chunk_size}"
        )

    def _initialize_vectorstore(self):
        """初始化向量存储"""
        with self._lock:
            if self.persist_directory:
                # 持久化模式
                persist_path = Path(self.persist_directory)
                persist_path.mkdir(parents=True, exist_ok=True)

                self.vectorstore = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=str(persist_path),
                )
                logger.info(f"Chroma initialized in persistent mode: {persist_path}")
            else:
                # 内存模式
                self.vectorstore = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                )
                logger.info("Chroma initialized in memory mode")

    def add_news(
        self,
        news_list: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """
        添加新闻到向量存储

        Args:
            news_list: 新闻列表，每条新闻包含 title, content, source 等字段
            batch_size: 批处理大小

        Returns:
            添加的文档数量
        """
        with self._lock:
            try:
                logger.info(f"Adding {len(news_list)} news items to vector store")

                # 转换为 Document 对象
                documents = []
                for news in news_list:
                    title = news.get("title", "")
                    content = news.get("content", "")
                    source = news.get("source", "")
                    timestamp = news.get("timestamp", "")
                    url = news.get("url", "")

                    # 跳过空内容
                    if not title or not content:
                        logger.warning(f"Skipping news with empty title or content: {news}")
                        continue

                    # 构建文档文本
                    text = f"标题: {title}\n\n内容: {content}"

                    # 构建元数据
                    metadata = {
                        "title": title,
                        "source": source,
                        "timestamp": timestamp,
                        "url": url,
                        "type": "news",
                    }

                    doc = Document(page_content=text, metadata=metadata)
                    documents.append(doc)

                if not documents:
                    logger.warning("No valid documents to add")
                    return 0

                # 分块处理
                chunks = self.text_splitter.split_documents(documents)
                logger.info(f"Split into {len(chunks)} chunks")

                # 批量添加
                total_added = 0
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    self.vectorstore.add_documents(batch)
                    total_added += len(batch)
                    logger.debug(f"Added batch {i // batch_size + 1}: {len(batch)} chunks")

                logger.info(f"Successfully added {total_added} chunks from {len(documents)} news items")

                return total_added

            except Exception as e:
                error_msg = f"Failed to add news to vector store: {str(e)}"
                logger.error(error_msg, exc_info=True)
                raise

    def search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        搜索相关文档

        Args:
            query: 查询文本
            k: 返回的文档数量
            filter_metadata: 元数据过滤条件

        Returns:
            相关文档列表
        """
        with self._lock:
            try:
                logger.info(f"Searching for: '{query}' (k={k})")

                if filter_metadata:
                    results = self.vectorstore.similarity_search(
                        query=query,
                        k=k,
                        filter=filter_metadata,
                    )
                else:
                    results = self.vectorstore.similarity_search(
                        query=query,
                        k=k,
                    )

                logger.info(f"Found {len(results)} relevant documents")

                return results

            except Exception as e:
                error_msg = f"Failed to search vector store: {str(e)}"
                logger.error(error_msg, exc_info=True)
                raise

    def search_with_score(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[tuple[Document, float]]:
        """
        搜索相关文档并返回相似度分数

        Args:
            query: 查询文本
            k: 返回的文档数量
            filter_metadata: 元数据过滤条件

        Returns:
            (文档, 分数) 元组列表
        """
        with self._lock:
            try:
                logger.info(f"Searching with score for: '{query}' (k={k})")

                if filter_metadata:
                    results = self.vectorstore.similarity_search_with_score(
                        query=query,
                        k=k,
                        filter=filter_metadata,
                    )
                else:
                    results = self.vectorstore.similarity_search_with_score(
                        query=query,
                        k=k,
                    )

                logger.info(f"Found {len(results)} documents with scores")

                return results

            except Exception as e:
                error_msg = f"Failed to search with score: {str(e)}"
                logger.error(error_msg, exc_info=True)
                raise

    def delete_collection(self):
        """删除整个集合"""
        with self._lock:
            try:
                logger.warning(f"Deleting collection: {self.collection_name}")
                self.vectorstore.delete_collection()
                logger.info("Collection deleted successfully")

                # 重新初始化
                self._initialize_vectorstore()

            except Exception as e:
                error_msg = f"Failed to delete collection: {str(e)}"
                logger.error(error_msg, exc_info=True)
                raise

    def get_stats(self) -> Dict[str, Any]:
        """
        获取向量存储统计信息

        Returns:
            统计信息字典
        """
        with self._lock:
            try:
                # Chroma 的统计信息
                collection = self.vectorstore._collection
                count = collection.count()

                return {
                    "collection_name": self.collection_name,
                    "document_count": count,
                    "persist_directory": self.persist_directory,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                }

            except Exception as e:
                logger.error(f"Failed to get stats: {e}", exc_info=True)
                return {
                    "collection_name": self.collection_name,
                    "error": str(e),
                }

    def as_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        转换为 LangChain Retriever

        Args:
            search_kwargs: 搜索参数（如 k, filter 等）

        Returns:
            Retriever 实例
        """
        if search_kwargs is None:
            search_kwargs = {"k": 5}

        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)


# ==================== 便捷函数 ====================


def create_news_vectorstore(
    persist_directory: Optional[str] = None,
    collection_name: str = "trendradar_news",
    chunk_size: int = 1000,
) -> NewsVectorStore:
    """
    创建新闻向量存储的工厂函数

    Args:
        persist_directory: 持久化目录
        collection_name: 集合名称
        chunk_size: 分块大小

    Returns:
        NewsVectorStore 实例
    """
    logger.info(f"Creating NewsVectorStore: {collection_name}")
    return NewsVectorStore(
        persist_directory=persist_directory,
        collection_name=collection_name,
        chunk_size=chunk_size,
    )


def get_default_vectorstore() -> NewsVectorStore:
    """
    获取默认的向量存储实例（使用项目根目录的 .chroma 目录）

    Returns:
        NewsVectorStore 实例
    """
    import os
    project_root = Path(__file__).parent.parent.parent
    persist_dir = project_root / ".chroma"

    return create_news_vectorstore(
        persist_directory=str(persist_dir),
        collection_name="trendradar_news",
    )
