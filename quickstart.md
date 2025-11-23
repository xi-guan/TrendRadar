# TrendRadar 快速启动指南 (macOS)

## 环境要求

- Python 3.10+
- uv (包管理器)

## 安装

```bash
# 1. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 克隆项目
git clone <repo-url>
cd TrendRadar

# 3. 安装依赖
uv sync --group web
```

## 配置

编辑 `config/config.yaml`:

```bash
vim config/config.yaml
```

关键配置：
- `crawler.enable_crawler: true` - 启用爬虫
- `platforms` - 选择要抓取的平台
- `notification.webhooks` - 通知渠道（可选）

## 启动

### 方式 1: Web 界面

```bash
./web/start_backend.sh
```

访问: http://localhost:8007

### 方式 2: 命令行爬虫

```bash
uv run python main.py
```

数据保存到: `data/YYYY-MM-DD/*.json`

## 可选功能

### AI 分析 (LangChain)

```bash
# 安装依赖
uv sync --group langchain

# 使用 Ollama 本地模型
ollama pull qwen2.5:14b
ollama pull nomic-embed-text

# 配置
./scripts/setup.sh
vim config/local.yaml  # 设置 provider: ollama
```

### 定时任务

```bash
# 添加到 crontab
crontab -e
```

```cron
0 */1 * * * cd /path/to/TrendRadar && uv run python main.py
```

## 故障排查

### 端口占用

```bash
lsof -ti:8007 | xargs kill -9
```

### 权限问题

```bash
chmod +x web/start_backend.sh
```

### 查看日志

```bash
tail -f logs/*.log
```
