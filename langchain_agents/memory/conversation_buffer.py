"""
Conversation Buffer Memory

实现基础对话缓冲记忆，用于多轮对话场景。
"""

import logging
import threading
from typing import List, Dict, Any, Optional
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage


logger = logging.getLogger(__name__)


class ConversationBufferMemory(BaseChatMessageHistory):
    """
    对话缓冲记忆

    功能：
    - 存储完整的对话历史
    - 支持添加/检索消息
    - 线程安全
    - 可选的最大消息数限制
    """

    def __init__(
        self,
        max_messages: Optional[int] = None,
        return_messages: bool = True,
    ):
        """
        初始化对话缓冲记忆

        Args:
            max_messages: 最大消息数（None 表示无限制）
            return_messages: 是否返回消息对象（True）或字符串（False）
        """
        self.max_messages = max_messages
        self.return_messages = return_messages

        # 消息存储
        self._messages: List[BaseMessage] = []

        # 线程锁 (从 P0 修复学到的经验)
        self._lock = threading.RLock()

        logger.info(
            f"ConversationBufferMemory initialized: "
            f"max_messages={max_messages}, return_messages={return_messages}"
        )

    @property
    def messages(self) -> List[BaseMessage]:
        """
        获取所有消息

        Returns:
            消息列表
        """
        with self._lock:
            return list(self._messages)  # Return copy for thread safety

    def add_message(self, message: BaseMessage) -> None:
        """
        添加一条消息

        Args:
            message: 消息对象
        """
        with self._lock:
            self._messages.append(message)

            # 如果超过最大限制，移除最旧的消息
            if self.max_messages is not None and len(self._messages) > self.max_messages:
                removed_count = len(self._messages) - self.max_messages
                self._messages = self._messages[removed_count:]
                logger.debug(f"Removed {removed_count} old messages (max_messages={self.max_messages})")

            logger.debug(f"Message added: {message.__class__.__name__} (total: {len(self._messages)})")

    def add_user_message(self, message: str) -> None:
        """
        添加用户消息

        Args:
            message: 用户消息文本
        """
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        """
        添加 AI 消息

        Args:
            message: AI 消息文本
        """
        self.add_message(AIMessage(content=message))

    def clear(self) -> None:
        """清空所有消息"""
        with self._lock:
            message_count = len(self._messages)
            self._messages.clear()
            logger.info(f"Memory cleared ({message_count} messages removed)")

    def get_messages(self) -> List[BaseMessage]:
        """
        获取所有消息（别名方法）

        Returns:
            消息列表
        """
        return self.messages

    def get_formatted_messages(self) -> str:
        """
        获取格式化的消息历史

        Returns:
            格式化的消息字符串
        """
        with self._lock:
            if not self._messages:
                return ""

            formatted = []
            for msg in self._messages:
                if isinstance(msg, HumanMessage):
                    formatted.append(f"Human: {msg.content}")
                elif isinstance(msg, AIMessage):
                    formatted.append(f"AI: {msg.content}")
                elif isinstance(msg, SystemMessage):
                    formatted.append(f"System: {msg.content}")
                else:
                    formatted.append(f"{msg.__class__.__name__}: {msg.content}")

            return "\n".join(formatted)

    def get_stats(self) -> Dict[str, Any]:
        """
        获取内存统计信息

        Returns:
            统计信息字典
        """
        with self._lock:
            message_types = {}
            for msg in self._messages:
                msg_type = msg.__class__.__name__
                message_types[msg_type] = message_types.get(msg_type, 0) + 1

            return {
                "total_messages": len(self._messages),
                "max_messages": self.max_messages,
                "message_types": message_types,
                "return_messages": self.return_messages,
            }

    def __len__(self) -> int:
        """返回消息数量"""
        with self._lock:
            return len(self._messages)

    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"ConversationBufferMemory("
            f"messages={len(self._messages)}, "
            f"max_messages={self.max_messages})"
        )


class ConversationBufferWindowMemory(ConversationBufferMemory):
    """
    对话缓冲窗口记忆

    与 ConversationBufferMemory 相同，但强制设置 max_messages。
    保留最近的 k 条消息。
    """

    def __init__(self, k: int = 10, return_messages: bool = True):
        """
        初始化窗口记忆

        Args:
            k: 窗口大小（保留最近 k 条消息）
            return_messages: 是否返回消息对象
        """
        if k <= 0:
            raise ValueError(f"Window size k must be positive, got {k}")

        super().__init__(max_messages=k, return_messages=return_messages)

        logger.info(f"ConversationBufferWindowMemory initialized with k={k}")

    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"ConversationBufferWindowMemory("
            f"messages={len(self._messages)}, "
            f"k={self.max_messages})"
        )


# ==================== 便捷函数 ====================


def create_memory(
    memory_type: str = "buffer",
    max_messages: Optional[int] = None,
    return_messages: bool = True,
) -> ConversationBufferMemory:
    """
    创建对话记忆的工厂函数

    Args:
        memory_type: 记忆类型 ("buffer", "window")
        max_messages: 最大消息数（仅用于 "buffer"）
        return_messages: 是否返回消息对象

    Returns:
        对话记忆实例
    """
    if memory_type == "buffer":
        return ConversationBufferMemory(
            max_messages=max_messages,
            return_messages=return_messages,
        )
    elif memory_type == "window":
        if max_messages is None:
            max_messages = 10  # Default window size
        return ConversationBufferWindowMemory(
            k=max_messages,
            return_messages=return_messages,
        )
    else:
        raise ValueError(
            f"Invalid memory_type: {memory_type}. "
            f"Must be one of ['buffer', 'window']"
        )
