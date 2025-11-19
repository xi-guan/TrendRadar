# TrendRadar LangChain 1.0 集成

> AI 增强功能 - 智能新闻分析、趋势预测、对话式交互

## 📋 概述

这是 TrendRadar 的 **可选 AI 增强模块**，基于 LangChain 1.0 构建，提供：

- ✅ **智能新闻摘要**：自动生成每日新闻精华
- ✅ **趋势预测分析**：预测热点话题的未来走势
- ✅ **语义搜索**：基于含义而非关键词的智能搜索
- ✅ **对话式交互**：用自然语言查询新闻

## 🚀 快速开始

### 安装依赖

```bash
# 安装 LangChain 依赖组
uv sync --group langchain

# 或者手动安装
pip install langchain>=1.0.0 langchain-openai langchain-community chromadb
```

### 环境变量配置

创建 `.env` 文件：

```bash
# OpenAI API Key（必需）
OPENAI_API_KEY=sk-xxx

# LangChain 配置（可选）
LANGCHAIN_PROVIDER=openai
LANGCHAIN_MODEL=gpt-4o-mini
LANGCHAIN_TEMPERATURE=0.3
LANGCHAIN_MAX_TOKENS=1000

# 成本控制（可选）
LANGCHAIN_MAX_COST_PER_DAY=10.0
LANGCHAIN_MAX_REQUESTS_PER_MINUTE=10
```

### 使用示例

#### 1. 新闻智能摘要

```python
from langchain_agents.agents.news_analyst import NewsAnalystAgent

# 创建 Agent
agent = NewsAnalystAgent()

# 生成每日摘要
summary = agent.analyze_daily_news()
print(summary)

# 对比平台
comparison = agent.compare_platforms("人工智能")
print(comparison)

# 预测趋势
prediction = agent.predict_trend("比特币")
print(prediction)
```

#### 2. 对话式交互

```python
from langchain_agents.agents.news_analyst import NewsAnalystAgent

agent = NewsAnalystAgent()

# 自由提问
result = agent.invoke({
    "input": "比较一下微博和知乎今天的科技热点有什么不同？"
})
print(result["output"])
```

#### 3. 自定义工具

```python
from langchain_agents.agents.base_agent import BaseAgent
from langchain_agents.tools.trendradar_tools import get_all_trendradar_tools
from langchain.agents import create_agent

# 获取默认工具
tools = get_all_trendradar_tools()

# 添加自定义工具
# tools.append(MyCustomTool())

# 创建自定义 Agent
class MyCustomAgent(BaseAgent):
    def __init__(self):
        super().__init__(tools=tools)
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt="你的自定义提示词"
        )
```

## 🛠️ 可用工具

### 1. GetLatestNewsTool
获取最新爬取的新闻数据

**参数**：
- `platforms`: 平台列表，如 `['zhihu', 'weibo']`
- `limit`: 返回数量，默认 50
- `include_url`: 是否包含链接

### 2. AnalyzeTrendTool
分析话题趋势

**参数**：
- `topic`: 话题关键词（必需）
- `analysis_type`: 分析类型
  - `trend`: 热度趋势
  - `lifecycle`: 生命周期
  - `viral`: 异常热度检测
  - `predict`: 话题预测
- `date_range`: 日期范围

### 3. SearchNewsTool
搜索相关新闻

**参数**：
- `keyword`: 搜索关键词（必需）
- `platforms`: 平台列表
- `limit`: 返回数量

## ⚙️ 配置说明

### LLM Provider

支持多种 LLM Provider：

```python
# OpenAI (默认)
LANGCHAIN_PROVIDER=openai
LANGCHAIN_MODEL=gpt-4o-mini

# Anthropic (计划支持)
LANGCHAIN_PROVIDER=anthropic
LANGCHAIN_MODEL=claude-3-5-sonnet-20241022

# Ollama (本地模型，计划支持)
LANGCHAIN_PROVIDER=ollama
LANGCHAIN_MODEL=llama2
```

### 成本控制

```python
from langchain_agents.config import get_config

config = get_config()
print(f"每日成本限制: ${config.cost.max_cost_per_day}")
print(f"每分钟请求限制: {config.rate_limit.max_requests_per_minute}")
```

## 📊 成本估算

使用 `gpt-4o-mini` 的成本参考：

| 使用场景 | 每日请求 | 月成本估算 |
|---------|---------|----------|
| 个人用户 | 10 次 | ~$1 |
| 小团队 | 100 次 | ~$5 |
| 企业 | 1000 次 | ~$50 |

**成本优化建议**：
- 启用缓存（默认开启，TTL 1小时）
- 使用 `gpt-4o-mini` 而非 `gpt-4o`
- 设置每日成本上限

## 🔧 高级功能

### 缓存中间件

自动缓存 LLM 响应，减少重复调用：

```python
config.cache.enabled = True
config.cache.ttl = 3600  # 1 小时
```

### 速率限制

防止 API 超限：

```python
config.rate_limit.enabled = True
config.rate_limit.max_requests_per_minute = 10
```

## 🐛 故障排查

### 问题 1: API Key 未设置

```
ValueError: OPENAI_API_KEY environment variable not set
```

**解决**：设置环境变量 `export OPENAI_API_KEY=sk-xxx`

### 问题 2: 依赖未安装

```
ModuleNotFoundError: No module named 'langchain'
```

**解决**：运行 `uv sync --group langchain`

### 问题 3: 成本超限

```
CostLimitExceeded: Daily cost limit exceeded
```

**解决**：调整 `LANGCHAIN_MAX_COST_PER_DAY` 或等待明天

## 📚 更多资源

- [LangChain 官方文档](https://python.langchain.com/)
- [TrendRadar 主文档](../readme.md)
- [示例代码](./examples/)

## 🤝 贡献

欢迎贡献新的 Agent 或工具！请提交 PR 到主仓库。

## 📄 许可证

与 TrendRadar 主项目相同，遵循 GPL-3.0 许可证。
