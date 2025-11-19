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
class EmbeddingsConfig:
    """Embeddings 配置"""

    provider: Literal["openai", "ollama"] = "openai"
    model: str = "text-embedding-3-small"
    base_url: Optional[str] = None


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
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
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

        # 根据 provider 设置默认模型
        default_models = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-5-sonnet-20241022",
            "ollama": "qwen2.5:14b",  # 推荐中文模型
        }

        llm_config = LLMProviderConfig(
            provider=provider,
            model=os.getenv("LANGCHAIN_MODEL", default_models.get(provider, "gpt-4o-mini")),
            api_key=os.getenv(
                f"{provider.upper()}_API_KEY",
                os.getenv("OPENAI_API_KEY") if provider == "openai" else None,
            ),
            base_url=os.getenv("LANGCHAIN_BASE_URL"),
            temperature=_parse_float("LANGCHAIN_TEMPERATURE", 0.3, min_val=0.0, max_val=2.0),
            max_tokens=_parse_int("LANGCHAIN_MAX_TOKENS", 1000, min_val=1, max_val=128000),
            timeout=_parse_int("LANGCHAIN_TIMEOUT", 60, min_val=1, max_val=600),
        )

        # Embeddings 配置
        embeddings_provider = os.getenv("LANGCHAIN_EMBEDDINGS_PROVIDER", "openai").lower()
        if embeddings_provider not in ["openai", "ollama"]:
            logger.warning(
                f"Invalid LANGCHAIN_EMBEDDINGS_PROVIDER={embeddings_provider}, using 'openai'"
            )
            embeddings_provider = "openai"

        # 根据 provider 设置默认 embeddings 模型
        default_embeddings_models = {
            "openai": "text-embedding-3-small",
            "ollama": "nomic-embed-text",  # Ollama 推荐的 embeddings 模型
        }

        embeddings_config = EmbeddingsConfig(
            provider=embeddings_provider,
            model=os.getenv(
                "LANGCHAIN_EMBEDDINGS_MODEL",
                default_embeddings_models.get(embeddings_provider, "text-embedding-3-small"),
            ),
            base_url=os.getenv("LANGCHAIN_EMBEDDINGS_BASE_URL"),
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
            embeddings=embeddings_config,
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


# ==================== LLM 工厂函数 ====================


def create_llm(config: Optional[LangChainConfig] = None):
    """
    根据配置创建 LLM 实例

    Args:
        config: LangChain 配置（None 则使用全局配置）

    Returns:
        LLM 实例
    """
    if config is None:
        config = get_config()

    provider = config.llm.provider
    model = config.llm.model
    base_url = config.llm.base_url
    temperature = config.llm.temperature
    max_tokens = config.llm.max_tokens
    timeout = config.llm.timeout

    logger.info(f"Creating LLM: provider={provider}, model={model}")

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=config.llm.api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        # Ollama 的 base_url 默认是 http://localhost:11434
        if base_url is None:
            base_url = "http://localhost:11434"

        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_predict=max_tokens,  # Ollama 使用 num_predict 代替 max_tokens
        )

    elif provider == "anthropic":
        # 未来可以添加 Anthropic 支持
        raise NotImplementedError(
            "Anthropic provider is not yet implemented. Please use 'openai' or 'ollama'."
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def create_embeddings(config: Optional[LangChainConfig] = None):
    """
    根据配置创建 Embeddings 实例

    Args:
        config: LangChain 配置（None 则使用全局配置）

    Returns:
        Embeddings 实例
    """
    if config is None:
        config = get_config()

    provider = config.embeddings.provider
    model = config.embeddings.model
    base_url = config.embeddings.base_url

    logger.info(f"Creating Embeddings: provider={provider}, model={model}")

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=model,
            base_url=base_url,
        )

    elif provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        # Ollama 的 base_url 默认是 http://localhost:11434
        if base_url is None:
            base_url = "http://localhost:11434"

        return OllamaEmbeddings(
            model=model,
            base_url=base_url,
        )

    else:
        raise ValueError(f"Unsupported embeddings provider: {provider}")
