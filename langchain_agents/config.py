"""
LangChain 配置管理

支持多 LLM Provider、环境变量管理、成本控制等
"""

import os
from typing import Literal, Optional
from dataclasses import dataclass, field


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
class CacheConfig:
    """缓存配置"""

    enabled: bool = True
    ttl: int = 3600  # 缓存过期时间（秒）
    max_size: int = 1000  # 最大缓存条目数


@dataclass
class RateLimitConfig:
    """速率限制配置"""

    enabled: bool = True
    max_requests_per_minute: int = 10
    max_tokens_per_minute: int = 50000


@dataclass
class CostConfig:
    """成本控制配置"""

    enabled: bool = True
    max_cost_per_day: float = 10.0  # 美元
    alert_threshold: float = 0.8  # 80% 时发出警告


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
    cache: CacheConfig = field(default_factory=CacheConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)

    @classmethod
    def from_env(cls) -> "LangChainConfig":
        """从环境变量加载配置"""
        # LLM Provider 配置
        provider = os.getenv("LANGCHAIN_PROVIDER", "openai").lower()
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
            temperature=float(os.getenv("LANGCHAIN_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("LANGCHAIN_MAX_TOKENS", "1000")),
            timeout=int(os.getenv("LANGCHAIN_TIMEOUT", "60")),
        )

        # 缓存配置
        cache_config = CacheConfig(
            enabled=os.getenv("LANGCHAIN_CACHE_ENABLED", "true").lower() == "true",
            ttl=int(os.getenv("LANGCHAIN_CACHE_TTL", "3600")),
            max_size=int(os.getenv("LANGCHAIN_CACHE_MAX_SIZE", "1000")),
        )

        # 速率限制配置
        rate_limit_config = RateLimitConfig(
            enabled=os.getenv("LANGCHAIN_RATE_LIMIT_ENABLED", "true").lower()
            == "true",
            max_requests_per_minute=int(
                os.getenv("LANGCHAIN_MAX_REQUESTS_PER_MINUTE", "10")
            ),
            max_tokens_per_minute=int(
                os.getenv("LANGCHAIN_MAX_TOKENS_PER_MINUTE", "50000")
            ),
        )

        # 成本控制配置
        cost_config = CostConfig(
            enabled=os.getenv("LANGCHAIN_COST_ENABLED", "true").lower() == "true",
            max_cost_per_day=float(os.getenv("LANGCHAIN_MAX_COST_PER_DAY", "10.0")),
            alert_threshold=float(os.getenv("LANGCHAIN_ALERT_THRESHOLD", "0.8")),
        )

        # 向量数据库配置
        vector_store_config = VectorStoreConfig(
            provider=os.getenv("LANGCHAIN_VECTOR_STORE", "chroma"),
            persist_directory=os.getenv(
                "LANGCHAIN_VECTOR_PERSIST_DIR", "./data/vectors"
            ),
            collection_name=os.getenv(
                "LANGCHAIN_VECTOR_COLLECTION", "trendradar_news"
            ),
        )

        return cls(
            llm=llm_config,
            cache=cache_config,
            rate_limit=rate_limit_config,
            cost=cost_config,
            vector_store=vector_store_config,
        )

    def validate(self) -> bool:
        """验证配置"""
        if self.llm.provider in ["openai", "anthropic"] and not self.llm.api_key:
            raise ValueError(
                f"{self.llm.provider.upper()}_API_KEY environment variable not set"
            )
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
