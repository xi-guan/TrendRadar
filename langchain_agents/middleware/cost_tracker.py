"""
Cost Tracker Middleware

跟踪和控制 LLM API 调用成本。
"""

import logging
import threading
import time
from collections import deque
from typing import Any, Dict, Optional, Callable
from functools import wraps
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


class CostLimitExceeded(Exception):
    """成本限制超出异常"""
    pass


# ==================== 定价表 ====================

# OpenAI GPT-4o-mini 定价 (2025年1月)
PRICING_TABLE = {
    "openai": {
        "gpt-4o-mini": {
            "input": 0.150 / 1_000_000,  # $0.150 per 1M input tokens
            "output": 0.600 / 1_000_000,  # $0.600 per 1M output tokens
        },
        "gpt-4o": {
            "input": 2.50 / 1_000_000,  # $2.50 per 1M input tokens
            "output": 10.00 / 1_000_000,  # $10.00 per 1M output tokens
        },
        "gpt-4-turbo": {
            "input": 10.00 / 1_000_000,
            "output": 30.00 / 1_000_000,
        },
        "gpt-3.5-turbo": {
            "input": 0.50 / 1_000_000,
            "output": 1.50 / 1_000_000,
        },
    },
    "anthropic": {
        "claude-3-5-sonnet-20241022": {
            "input": 3.00 / 1_000_000,
            "output": 15.00 / 1_000_000,
        },
        "claude-3-5-haiku-20241022": {
            "input": 0.80 / 1_000_000,
            "output": 4.00 / 1_000_000,
        },
        "claude-3-opus-20240229": {
            "input": 15.00 / 1_000_000,
            "output": 75.00 / 1_000_000,
        },
    },
}


class CostTrackerMiddleware:
    """
    成本跟踪中间件

    功能：
    - 实时跟踪 API 调用成本
    - 每日/每月成本限制
    - 成本预算预警
    - 详细的成本统计
    """

    def __init__(
        self,
        max_cost_per_day: float = 10.0,
        max_cost_per_month: float = 300.0,
        alert_threshold: float = 0.8,
        enabled: bool = True,
        pricing_table: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化成本跟踪中间件

        Args:
            max_cost_per_day: 每日最大成本（美元）
            max_cost_per_month: 每月最大成本（美元）
            alert_threshold: 预警阈值（0.8 表示达到 80% 时发出警告）
            enabled: 是否启用成本跟踪
            pricing_table: 自定义定价表（可选）
        """
        self.max_cost_per_day = max_cost_per_day
        self.max_cost_per_month = max_cost_per_month
        self.alert_threshold = alert_threshold
        self.enabled = enabled
        self.pricing_table = pricing_table or PRICING_TABLE

        # 成本记录: [(timestamp, cost, model, input_tokens, output_tokens), ...]
        # P1 修复: 使用 deque 限制大小防止内存泄漏
        self._cost_records = deque(maxlen=10000)

        # 线程锁 (P0 修复: 线程安全)
        self._lock = threading.RLock()

        # 统计信息
        self._total_cost = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._alert_sent = False

        logger.info(
            f"CostTrackerMiddleware initialized: "
            f"max_cost_per_day=${max_cost_per_day}, "
            f"max_cost_per_month=${max_cost_per_month}, "
            f"enabled={enabled} (max_records=10000)"
        )

    def _calculate_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        计算单次调用成本

        Args:
            provider: 提供商 (openai, anthropic)
            model: 模型名称
            input_tokens: 输入 token 数
            output_tokens: 输出 token 数

        Returns:
            成本（美元）
        """
        try:
            pricing = self.pricing_table.get(provider, {}).get(model, {})

            if not pricing:
                logger.warning(
                    f"No pricing info for {provider}/{model}, using default rate"
                )
                # 默认使用 GPT-4o-mini 的定价
                pricing = self.pricing_table["openai"]["gpt-4o-mini"]

            input_cost = input_tokens * pricing["input"]
            output_cost = output_tokens * pricing["output"]

            total_cost = input_cost + output_cost

            logger.debug(
                f"Cost calculated: ${total_cost:.6f} "
                f"({input_tokens} input + {output_tokens} output tokens)"
            )

            return total_cost

        except Exception as e:
            logger.error(f"Failed to calculate cost: {e}", exc_info=True)
            return 0.0

    def _get_period_cost(self, hours: int) -> float:
        """
        获取指定时间段内的成本

        Args:
            hours: 时间段（小时）

        Returns:
            时间段内的总成本
        """
        cutoff_time = time.time() - (hours * 3600)

        period_cost = sum(
            cost for timestamp, cost, _, _, _ in self._cost_records
            if timestamp >= cutoff_time
        )

        return period_cost

    def track(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        跟踪一次 API 调用的成本

        Args:
            provider: 提供商
            model: 模型名称
            input_tokens: 输入 token 数
            output_tokens: 输出 token 数

        Returns:
            本次调用成本

        Raises:
            CostLimitExceeded: 如果超出成本限制

        Note:
            P0 修复: 添加线程锁保证线程安全
        """
        if not self.enabled:
            return 0.0

        # 计算成本
        cost = self._calculate_cost(provider, model, input_tokens, output_tokens)

        with self._lock:
            # 检查每日限制
            daily_cost = self._get_period_cost(24) + cost
            if daily_cost > self.max_cost_per_day:
                raise CostLimitExceeded(
                    f"Daily cost limit exceeded: ${daily_cost:.4f} > ${self.max_cost_per_day}"
                )

            # 检查每月限制
            monthly_cost = self._get_period_cost(24 * 30) + cost
            if monthly_cost > self.max_cost_per_month:
                raise CostLimitExceeded(
                    f"Monthly cost limit exceeded: ${monthly_cost:.4f} > ${self.max_cost_per_month}"
                )

            # 检查预警阈值
            if not self._alert_sent:
                if daily_cost >= self.max_cost_per_day * self.alert_threshold:
                    logger.warning(
                        f"⚠️  Daily cost approaching limit: "
                        f"${daily_cost:.4f} / ${self.max_cost_per_day} "
                        f"({daily_cost / self.max_cost_per_day * 100:.1f}%)"
                    )
                    self._alert_sent = True

            # 记录成本
            self._cost_records.append((
                time.time(),
                cost,
                model,
                input_tokens,
                output_tokens,
            ))

            # 更新总计
            self._total_cost += cost
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens

            logger.info(
                f"Cost tracked: ${cost:.6f} ({model}, "
                f"{input_tokens} input + {output_tokens} output tokens)"
            )

            return cost

    def get_stats(self, period: str = "all") -> Dict[str, Any]:
        """
        获取成本统计信息

        Args:
            period: 统计周期 ("all", "day", "week", "month")

        Returns:
            统计信息字典

        Note:
            P0 修复: 添加线程锁保证线程安全
        """
        with self._lock:
            # 确定时间范围
            if period == "day":
                period_hours = 24
            elif period == "week":
                period_hours = 24 * 7
            elif period == "month":
                period_hours = 24 * 30
            else:  # "all"
                period_hours = None

            # 过滤记录
            if period_hours:
                cutoff_time = time.time() - (period_hours * 3600)
                records = [
                    (timestamp, cost, model, input_tokens, output_tokens)
                    for timestamp, cost, model, input_tokens, output_tokens in self._cost_records
                    if timestamp >= cutoff_time
                ]
            else:
                records = list(self._cost_records)  # Create a copy for thread safety

            # 计算统计信息
            total_cost = sum(cost for _, cost, _, _, _ in records)
            total_input_tokens = sum(input_tokens for _, _, _, input_tokens, _ in records)
            total_output_tokens = sum(output_tokens for _, _, _, _, output_tokens in records)
            total_requests = len(records)

            # 按模型分组
            model_costs = {}
            for _, cost, model, _, _ in records:
                if model not in model_costs:
                    model_costs[model] = 0.0
                model_costs[model] += cost

            return {
                "enabled": self.enabled,
                "period": period,
                "total_cost": total_cost,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_requests": total_requests,
                "average_cost_per_request": total_cost / total_requests if total_requests > 0 else 0.0,
                "model_costs": model_costs,
                "limits": {
                    "max_cost_per_day": self.max_cost_per_day,
                    "max_cost_per_month": self.max_cost_per_month,
                    "daily_usage": self._get_period_cost(24),
                    "monthly_usage": self._get_period_cost(24 * 30),
                },
            }

    def reset_alert(self):
        """
        重置预警状态

        Note:
            P0 修复: 添加线程锁保证线程安全
        """
        with self._lock:
            self._alert_sent = False
            logger.info("Cost alert reset")

    def clear(self):
        """
        清空所有记录

        Note:
            P0 修复: 添加线程锁保证线程安全
        """
        with self._lock:
            self._cost_records.clear()
            self._total_cost = 0.0
            self._total_input_tokens = 0
            self._total_output_tokens = 0
            self._alert_sent = False
            logger.info("Cost tracker cleared")


# ==================== 装饰器 ====================


def cost_tracked(
    provider: str,
    model: str,
    max_cost_per_day: float = 10.0,
    max_cost_per_month: float = 300.0,
    enabled: bool = True,
    extract_usage: Optional[Callable] = None,
):
    """
    成本跟踪装饰器

    Usage:
        @cost_tracked(provider="openai", model="gpt-4o-mini")
        def llm_call(prompt):
            ...

    Args:
        provider: 提供商名称
        model: 模型名称
        max_cost_per_day: 每日最大成本
        max_cost_per_month: 每月最大成本
        enabled: 是否启用成本跟踪
        extract_usage: 从结果提取 usage 的函数

    Returns:
        装饰后的函数
    """
    tracker = CostTrackerMiddleware(
        max_cost_per_day=max_cost_per_day,
        max_cost_per_month=max_cost_per_month,
        enabled=enabled,
    )

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 执行函数
            result = func(*args, **kwargs)

            # 提取 token 使用情况
            if extract_usage is not None:
                try:
                    input_tokens, output_tokens = extract_usage(result)
                except Exception as e:
                    logger.warning(f"Failed to extract usage: {e}")
                    return result
            elif isinstance(result, dict) and "usage" in result:
                # 默认从 result["usage"] 提取
                usage = result["usage"]
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
            else:
                logger.warning("Cannot extract token usage from result")
                return result

            # 跟踪成本
            tracker.track(provider, model, input_tokens, output_tokens)

            return result

        # 附加统计方法
        wrapper.cost_stats = lambda period="all": tracker.get_stats(period)
        wrapper.cost_clear = tracker.clear

        return wrapper

    return decorator


# ==================== 全局成本跟踪实例 ====================


_global_cost_tracker: Optional[CostTrackerMiddleware] = None


def get_global_cost_tracker() -> CostTrackerMiddleware:
    """
    获取全局成本跟踪实例

    Returns:
        全局成本跟踪中间件实例
    """
    global _global_cost_tracker

    if _global_cost_tracker is None:
        # 从环境变量读取配置
        import os
        max_cost_per_day = float(os.getenv("LANGCHAIN_MAX_COST_PER_DAY", "10.0"))
        max_cost_per_month = float(os.getenv("LANGCHAIN_MAX_COST_PER_MONTH", "300.0"))
        alert_threshold = float(os.getenv("LANGCHAIN_COST_ALERT_THRESHOLD", "0.8"))
        enabled = os.getenv("LANGCHAIN_COST_ENABLED", "true").lower() == "true"

        _global_cost_tracker = CostTrackerMiddleware(
            max_cost_per_day=max_cost_per_day,
            max_cost_per_month=max_cost_per_month,
            alert_threshold=alert_threshold,
            enabled=enabled,
        )
        logger.info("Global cost tracker instance created")

    return _global_cost_tracker
