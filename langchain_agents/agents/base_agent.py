"""
基础 Agent 类

提供通用的 Agent 功能，所有具体 Agent 继承此类
"""

from typing import List, Optional, Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from ..config import get_config, create_llm


class BaseAgent:
    """基础 Agent 类"""

    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None,
        llm: Optional[BaseChatModel] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        初始化 Agent

        Args:
            tools: Agent 可用的工具列表
            llm: 语言模型，不提供则使用配置中的默认模型
            system_prompt: 系统提示词
        """
        self.tools = tools or []
        self.config = get_config()

        # 初始化 LLM
        if llm:
            self.llm = llm
        else:
            self.llm = self._create_default_llm()

        self.system_prompt = system_prompt or self._get_default_system_prompt()

    def _create_default_llm(self) -> BaseChatModel:
        """创建默认的 LLM"""
        return create_llm(self.config)

    def _get_default_system_prompt(self) -> str:
        """获取默认的系统提示词"""
        return "You are a helpful AI assistant."

    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        调用 Agent（需要子类实现）

        Args:
            input_data: 输入数据

        Returns:
            Agent 的输出结果
        """
        raise NotImplementedError("Subclass must implement invoke() method")
