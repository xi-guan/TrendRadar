"""
LangChain Memory 模块

实现对话记忆管理：
- ConversationBufferMemory: 基础对话缓冲记忆
- ConversationBufferWindowMemory: 窗口记忆（保留最近 k 条消息）
- ConversationSummaryMemory: 对话摘要记忆 (未来实现)
"""

from langchain_agents.memory.conversation_buffer import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    create_memory,
)

__all__ = [
    "ConversationBufferMemory",
    "ConversationBufferWindowMemory",
    "create_memory",
]
