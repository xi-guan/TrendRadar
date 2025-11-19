"""
Rate Limit Middleware

实现请求速率限制，防止超出 API 配额。
"""

import logging
import time
from collections import deque
from typing import Any, Dict, Optional, Callable
from functools import wraps


logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """速率限制超出异常"""
    pass


class RateLimitMiddleware:
    """
    速率限制中间件

    功能：
    - 滑动窗口速率限制
    - 每分钟请求数限制
    - 每分钟 token 数限制
    - 自动等待和重试
    """

    def __init__(
        self,
        max_requests_per_minute: int = 60,
        max_tokens_per_minute: int = 50000,
        enabled: bool = True,
        auto_wait: bool = True,
    ):
        """
        初始化速率限制中间件

        Args:
            max_requests_per_minute: 每分钟最大请求数
            max_tokens_per_minute: 每分钟最大 token 数
            enabled: 是否启用速率限制
            auto_wait: 是否自动等待（如果超限则等待，否则抛出异常）
        """
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.enabled = enabled
        self.auto_wait = auto_wait

        # 滑动窗口：记录每次请求的时间戳和 token 数
        # deque 用于高效的 append 和 popleft 操作
        self._request_times: deque[float] = deque()
        self._token_counts: deque[int] = deque()

        # 统计信息
        self._total_requests = 0
        self._total_tokens = 0
        self._total_waits = 0
        self._total_wait_time = 0.0

        logger.info(
            f"RateLimitMiddleware initialized: "
            f"max_requests_per_minute={max_requests_per_minute}, "
            f"max_tokens_per_minute={max_tokens_per_minute}, "
            f"enabled={enabled}, auto_wait={auto_wait}"
        )

    def _clean_old_records(self, current_time: float):
        """
        清理 1 分钟之前的记录

        Args:
            current_time: 当前时间戳
        """
        one_minute_ago = current_time - 60.0

        # 移除旧记录
        while self._request_times and self._request_times[0] < one_minute_ago:
            self._request_times.popleft()
            self._token_counts.popleft()

    def _get_current_usage(self, current_time: float) -> Dict[str, int]:
        """
        获取当前 1 分钟内的使用情况

        Args:
            current_time: 当前时间戳

        Returns:
            使用情况字典 {"requests": int, "tokens": int}
        """
        self._clean_old_records(current_time)

        return {
            "requests": len(self._request_times),
            "tokens": sum(self._token_counts),
        }

    def _calculate_wait_time(self, current_time: float, tokens: int) -> float:
        """
        计算需要等待的时间

        Args:
            current_time: 当前时间戳
            tokens: 本次请求的 token 数

        Returns:
            需要等待的秒数（0 表示无需等待）
        """
        usage = self._get_current_usage(current_time)

        wait_time = 0.0

        # 检查请求数限制
        if usage["requests"] >= self.max_requests_per_minute:
            # 需要等到最早的请求超过 1 分钟
            oldest_time = self._request_times[0]
            wait_for_request = oldest_time + 60.0 - current_time
            wait_time = max(wait_time, wait_for_request)

        # 检查 token 数限制
        if usage["tokens"] + tokens > self.max_tokens_per_minute:
            # 需要等到足够的 token 配额释放
            # 简化：等到最早的请求超过 1 分钟
            if self._request_times:
                oldest_time = self._request_times[0]
                wait_for_tokens = oldest_time + 60.0 - current_time
                wait_time = max(wait_time, wait_for_tokens)

        return max(0.0, wait_time)

    def acquire(self, tokens: int = 0):
        """
        获取速率限制许可

        Args:
            tokens: 本次请求预计使用的 token 数

        Raises:
            RateLimitExceeded: 如果速率限制超出且 auto_wait=False
        """
        if not self.enabled:
            return

        current_time = time.time()

        wait_time = self._calculate_wait_time(current_time, tokens)

        if wait_time > 0:
            if self.auto_wait:
                logger.warning(
                    f"Rate limit approached, waiting {wait_time:.2f}s "
                    f"(requests: {len(self._request_times)}/{self.max_requests_per_minute}, "
                    f"tokens: {sum(self._token_counts)}/{self.max_tokens_per_minute})"
                )
                time.sleep(wait_time)
                self._total_waits += 1
                self._total_wait_time += wait_time
                current_time = time.time()  # 更新时间
            else:
                usage = self._get_current_usage(current_time)
                raise RateLimitExceeded(
                    f"Rate limit exceeded: "
                    f"requests={usage['requests']}/{self.max_requests_per_minute}, "
                    f"tokens={usage['tokens']}/{self.max_tokens_per_minute}"
                )

        # 记录本次请求
        self._request_times.append(current_time)
        self._token_counts.append(tokens)
        self._total_requests += 1
        self._total_tokens += tokens

    def release(self, actual_tokens: Optional[int] = None):
        """
        释放速率限制（更新实际使用的 token 数）

        Args:
            actual_tokens: 实际使用的 token 数（如果与预估不同）
        """
        if not self.enabled or actual_tokens is None:
            return

        # 更新最后一次请求的 token 数
        if self._token_counts:
            estimated_tokens = self._token_counts[-1]
            self._token_counts[-1] = actual_tokens

            # 更新总计
            self._total_tokens = self._total_tokens - estimated_tokens + actual_tokens

    def get_stats(self) -> Dict[str, Any]:
        """
        获取速率限制统计信息

        Returns:
            统计信息字典
        """
        current_time = time.time()
        usage = self._get_current_usage(current_time)

        return {
            "enabled": self.enabled,
            "max_requests_per_minute": self.max_requests_per_minute,
            "max_tokens_per_minute": self.max_tokens_per_minute,
            "current_requests": usage["requests"],
            "current_tokens": usage["tokens"],
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "total_waits": self._total_waits,
            "total_wait_time": self._total_wait_time,
            "auto_wait": self.auto_wait,
        }

    def reset(self):
        """重置所有统计和记录"""
        self._request_times.clear()
        self._token_counts.clear()
        self._total_requests = 0
        self._total_tokens = 0
        self._total_waits = 0
        self._total_wait_time = 0.0
        logger.info("Rate limiter reset")


# ==================== 装饰器 ====================


def rate_limited(
    max_requests_per_minute: int = 60,
    max_tokens_per_minute: int = 50000,
    enabled: bool = True,
    auto_wait: bool = True,
    estimate_tokens: Optional[Callable] = None,
):
    """
    速率限制装饰器

    Usage:
        @rate_limited(max_requests_per_minute=60)
        def api_call(arg1, arg2):
            ...

    Args:
        max_requests_per_minute: 每分钟最大请求数
        max_tokens_per_minute: 每分钟最大 token 数
        enabled: 是否启用速率限制
        auto_wait: 是否自动等待
        estimate_tokens: Token 估算函数（接收函数参数，返回预估 token 数）

    Returns:
        装饰后的函数
    """
    limiter = RateLimitMiddleware(
        max_requests_per_minute=max_requests_per_minute,
        max_tokens_per_minute=max_tokens_per_minute,
        enabled=enabled,
        auto_wait=auto_wait,
    )

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 估算 token 数
            tokens = 0
            if estimate_tokens is not None:
                try:
                    tokens = estimate_tokens(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Failed to estimate tokens: {e}")

            # 获取许可
            limiter.acquire(tokens)

            # 执行函数
            result = func(*args, **kwargs)

            # 如果结果包含实际 token 数，更新记录
            if isinstance(result, dict) and "usage" in result:
                actual_tokens = result["usage"].get("total_tokens")
                limiter.release(actual_tokens)

            return result

        # 附加统计方法
        wrapper.rate_limit_stats = limiter.get_stats
        wrapper.rate_limit_reset = limiter.reset

        return wrapper

    return decorator


# ==================== 全局速率限制实例 ====================


_global_rate_limiter: Optional[RateLimitMiddleware] = None


def get_global_rate_limiter() -> RateLimitMiddleware:
    """
    获取全局速率限制实例

    Returns:
        全局速率限制中间件实例
    """
    global _global_rate_limiter

    if _global_rate_limiter is None:
        # 从环境变量读取配置
        import os
        max_requests = int(os.getenv("LANGCHAIN_MAX_REQUESTS_PER_MINUTE", "60"))
        max_tokens = int(os.getenv("LANGCHAIN_MAX_TOKENS_PER_MINUTE", "50000"))
        enabled = os.getenv("LANGCHAIN_RATE_LIMIT_ENABLED", "true").lower() == "true"
        auto_wait = os.getenv("LANGCHAIN_RATE_LIMIT_AUTO_WAIT", "true").lower() == "true"

        _global_rate_limiter = RateLimitMiddleware(
            max_requests_per_minute=max_requests,
            max_tokens_per_minute=max_tokens,
            enabled=enabled,
            auto_wait=auto_wait,
        )
        logger.info("Global rate limiter instance created")

    return _global_rate_limiter
