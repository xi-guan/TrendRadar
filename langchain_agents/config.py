"""
LangChain 配置管理

支持多 LLM Provider、环境变量管理、成本控制等
"""

import os
import logging
from typing import Literal, Optional
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


def _parse_float(
    key: str, default: float, min_val: Optional[float] = None, max_val: Optional[float] = None
) -> float:
    """
    安全地解析浮点数环境变量

    Args:
        key: 环境变量名
        default: 默认值
        min_val: 最小值限制
        max_val: 最大值限制

    Returns:
        解析后的浮点数
    """
    try:
        value = float(os.getenv(key, str(default)))

        # 验证范围
        if min_val is not None and value < min_val:
            logger.warning(f"{key}={value} is below minimum {min_val}, using minimum")
            return min_val
        if max_val is not None and value > max_val:
            logger.warning(f"{key}={value} exceeds maximum {max_val}, using maximum")
            return max_val

        return value

    except ValueError as e:
        logger.warning(
            f"Invalid value for {key}: {os.getenv(key)}, using default {default}. Error: {e}"
        )
        return default


def _parse_int(
    key: str, default: int, min_val: Optional[int] = None, max_val: Optional[int] = None
) -> int:
    """
    安全地解析整数环境变量

    Args:
        key: 环境变量名
        default: 默认值
        min_val: 最小值限制
        max_val: 最大值限制

    Returns:
        解析后的整数
    """
    try:
        value = int(os.getenv(key, str(default)))

        # 验证范围
        if min_val is not None and value < min_val:
            logger.warning(f"{key}={value} is below minimum {min_val}, using minimum")
            return min_val
        if max_val is not None and value > max_val:
            logger.warning(f"{key}={value} exceeds maximum {max_val}, using maximum")
            return max_val

        return value

    except ValueError as e:
        logger.warning(
            f"Invalid value for {key}: {os.getenv(key)}, using default {default}. Error: {e}"
        )
        return default


def _parse_bool(key: str, default: bool) -> bool:
    """
    安全地解析布尔值环境变量

    Args:
        key: 环境变量名
        default: 默认值

    Returns:
        解析后的布尔值
    """
    value = os.getenv(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


@dataclass
class LLMProviderConfig:
    """LLM Provider 配置"""

    provider: Literal["openai", "anthropic", "ollama"] = "openai"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 1000
    timeout: int = 60


@dataclass
class VectorStoreConfig:
    """向量数据库配置"""

    provider: Literal["chroma", "faiss"] = "chroma"
    persist_directory: str = "./data/vectors"
    collection_name: str = "trendradar_news"


@dataclass
class LangChainConfig:
    """LangChain 全局配置"""

    llm: LLMProviderConfig = field(default_factory=LLMProviderConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)

    @classmethod
    def from_env(cls) -> "LangChainConfig":
        """从环境变量加载配置"""
        # LLM Provider 配置
        provider = os.getenv("LANGCHAIN_PROVIDER", "openai").lower()

        # 验证 provider 值
        valid_providers = ["openai", "anthropic", "ollama"]
        if provider not in valid_providers:
            logger.warning(
                f"Invalid LANGCHAIN_PROVIDER={provider}, must be one of {valid_providers}. Using 'openai'"
            )
            provider = "openai"

        llm_config = LLMProviderConfig(
            provider=provider,
            model=os.getenv(
                "LANGCHAIN_MODEL",
                "gpt-4o-mini" if provider == "openai" else "claude-3-5-sonnet-20241022",
            ),
            api_key=os.getenv(
                f"{provider.upper()}_API_KEY",
                os.getenv("OPENAI_API_KEY") if provider == "openai" else None,
            ),
            base_url=os.getenv("LANGCHAIN_BASE_URL"),
            temperature=_parse_float("LANGCHAIN_TEMPERATURE", 0.3, min_val=0.0, max_val=2.0),
            max_tokens=_parse_int("LANGCHAIN_MAX_TOKENS", 1000, min_val=1, max_val=128000),
            timeout=_parse_int("LANGCHAIN_TIMEOUT", 60, min_val=1, max_val=600),
        )

        # 向量数据库配置
        vector_store_provider = os.getenv("LANGCHAIN_VECTOR_STORE", "chroma").lower()
        if vector_store_provider not in ["chroma", "faiss"]:
            logger.warning(
                f"Invalid LANGCHAIN_VECTOR_STORE={vector_store_provider}, using 'chroma'"
            )
            vector_store_provider = "chroma"

        vector_store_config = VectorStoreConfig(
            provider=vector_store_provider,
            persist_directory=os.getenv(
                "LANGCHAIN_VECTOR_PERSIST_DIR", "./data/vectors"
            ),
            collection_name=os.getenv(
                "LANGCHAIN_VECTOR_COLLECTION", "trendradar_news"
            ),
        )

        return cls(
            llm=llm_config,
            vector_store=vector_store_config,
        )

    def validate(self) -> bool:
        """验证配置"""
        if self.llm.provider in ["openai", "anthropic"] and not self.llm.api_key:
            raise ValueError(
                f"{self.llm.provider.upper()}_API_KEY environment variable not set. "
                f"Please set OPENAI_API_KEY or ANTHROPIC_API_KEY."
            )

        logger.info(f"Configuration validated: provider={self.llm.provider}, model={self.llm.model}")
        return True


# 全局配置实例（延迟加载）
_global_config: Optional[LangChainConfig] = None


def get_config() -> LangChainConfig:
    """获取全局配置"""
    global _global_config
    if _global_config is None:
        _global_config = LangChainConfig.from_env()
        _global_config.validate()
    return _global_config


def reset_config():
    """重置全局配置（主要用于测试）"""
    global _global_config
    _global_config = None
