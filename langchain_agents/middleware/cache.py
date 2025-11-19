"""
Cache Middleware

实现 LLM 响应缓存，减少 API 调用和成本。
"""

import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional, Callable
from functools import wraps


logger = logging.getLogger(__name__)


class CacheMiddleware:
    """
    缓存中间件

    功能：
    - 基于输入哈希的缓存
    - TTL (Time To Live) 过期机制
    - 最大缓存大小限制
    - 缓存命中率统计
    """

    def __init__(
        self,
        ttl: int = 3600,
        max_size: int = 1000,
        enabled: bool = True,
    ):
        """
        初始化缓存中间件

        Args:
            ttl: 缓存过期时间（秒），默认 1 小时
            max_size: 最大缓存条目数，默认 1000
            enabled: 是否启用缓存，默认 True
        """
        self.ttl = ttl
        self.max_size = max_size
        self.enabled = enabled

        # 缓存存储: {cache_key: (result, timestamp)}
        self._cache: Dict[str, tuple[Any, float]] = {}

        # 统计信息
        self._hits = 0
        self._misses = 0

        logger.info(
            f"CacheMiddleware initialized: ttl={ttl}s, max_size={max_size}, enabled={enabled}"
        )

    def _generate_cache_key(self, *args, **kwargs) -> str:
        """
        生成缓存键

        Args:
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            缓存键（SHA256 哈希）
        """
        # 将参数序列化为字符串
        key_data = {
            "args": args,
            "kwargs": kwargs,
        }

        try:
            key_str = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            # 如果无法序列化，使用 repr
            key_str = repr(key_data)

        # 生成哈希
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _is_expired(self, timestamp: float) -> bool:
        """
        检查缓存是否过期

        Args:
            timestamp: 缓存时间戳

        Returns:
            是否过期
        """
        return time.time() - timestamp > self.ttl

    def _evict_expired(self):
        """清理过期缓存"""
        expired_keys = [
            key for key, (_, timestamp) in self._cache.items()
            if self._is_expired(timestamp)
        ]

        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            logger.debug(f"Evicted {len(expired_keys)} expired cache entries")

    def _evict_lru(self):
        """清理最旧的缓存（LRU）"""
        if len(self._cache) >= self.max_size:
            # 按时间戳排序，删除最旧的
            sorted_items = sorted(
                self._cache.items(),
                key=lambda x: x[1][1]  # 按 timestamp 排序
            )

            # 删除最旧的 10%
            evict_count = max(1, self.max_size // 10)
            for key, _ in sorted_items[:evict_count]:
                del self._cache[key]

            logger.debug(f"Evicted {evict_count} LRU cache entries")

    def get(self, *args, **kwargs) -> Optional[Any]:
        """
        从缓存获取结果

        Args:
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            缓存的结果，如果未命中则返回 None
        """
        if not self.enabled:
            return None

        cache_key = self._generate_cache_key(*args, **kwargs)

        if cache_key in self._cache:
            result, timestamp = self._cache[cache_key]

            if self._is_expired(timestamp):
                # 过期，删除
                del self._cache[cache_key]
                self._misses += 1
                logger.debug(f"Cache expired for key {cache_key[:16]}...")
                return None

            # 命中
            self._hits += 1
            logger.debug(f"Cache hit for key {cache_key[:16]}...")
            return result

        # 未命中
        self._misses += 1
        logger.debug(f"Cache miss for key {cache_key[:16]}...")
        return None

    def set(self, result: Any, *args, **kwargs):
        """
        设置缓存

        Args:
            result: 要缓存的结果
            *args: 位置参数
            **kwargs: 关键字参数
        """
        if not self.enabled:
            return

        # 清理过期缓存
        self._evict_expired()

        # 检查大小限制
        self._evict_lru()

        cache_key = self._generate_cache_key(*args, **kwargs)
        self._cache[cache_key] = (result, time.time())

        logger.debug(f"Cached result for key {cache_key[:16]}...")

    def clear(self):
        """清空所有缓存"""
        self._cache.clear()
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            统计信息字典
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return {
            "enabled": self.enabled,
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "cache_size": len(self._cache),
            "max_size": self.max_size,
            "ttl": self.ttl,
        }


# ==================== 装饰器 ====================


def cached(
    ttl: int = 3600,
    max_size: int = 1000,
    enabled: bool = True,
):
    """
    缓存装饰器

    Usage:
        @cached(ttl=3600, max_size=1000)
        def expensive_function(arg1, arg2):
            ...

    Args:
        ttl: 缓存过期时间（秒）
        max_size: 最大缓存大小
        enabled: 是否启用缓存

    Returns:
        装饰后的函数
    """
    cache = CacheMiddleware(ttl=ttl, max_size=max_size, enabled=enabled)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 尝试从缓存获取
            result = cache.get(*args, **kwargs)

            if result is not None:
                return result

            # 缓存未命中，执行函数
            result = func(*args, **kwargs)

            # 缓存结果
            cache.set(result, *args, **kwargs)

            return result

        # 附加统计方法
        wrapper.cache_stats = cache.get_stats
        wrapper.cache_clear = cache.clear

        return wrapper

    return decorator


# ==================== 全局缓存实例 ====================


_global_cache: Optional[CacheMiddleware] = None


def get_global_cache() -> CacheMiddleware:
    """
    获取全局缓存实例

    Returns:
        全局缓存中间件实例
    """
    global _global_cache

    if _global_cache is None:
        # 从环境变量读取配置
        import os
        ttl = int(os.getenv("LANGCHAIN_CACHE_TTL", "3600"))
        max_size = int(os.getenv("LANGCHAIN_CACHE_MAX_SIZE", "1000"))
        enabled = os.getenv("LANGCHAIN_CACHE_ENABLED", "true").lower() == "true"

        _global_cache = CacheMiddleware(ttl=ttl, max_size=max_size, enabled=enabled)
        logger.info("Global cache instance created")

    return _global_cache
