# Ollama 本地模型配置指南

本指南帮助您使用 Ollama 运行 TrendRadar 的本地大模型，实现完全离线、免费的 AI 功能。

## 为什么选择 Ollama？

- ✅ **完全免费**：无需 API Key，无使用限制
- ✅ **隐私保护**：数据完全本地处理，不上传云端
- ✅ **中文优化**：支持 Qwen、ChatGLM 等专为中文优化的模型
- ✅ **简单易用**：一键安装，命令行管理模型
- ✅ **性能优秀**：支持 GPU 加速，推理速度快

## 1. 安装 Ollama

### macOS / Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows
下载安装包：https://ollama.com/download/windows

### 验证安装
```bash
ollama --version
```

## 2. 下载中文模型

### 推荐模型（按中文能力排序）

| 模型 | 命令 | 参数量 | 内存需求 | 中文能力 | 推荐用途 |
|------|------|--------|----------|----------|----------|
| **Qwen2.5** | `ollama pull qwen2.5:14b` | 14B | 16GB | ⭐⭐⭐⭐⭐ | 综合最佳，强烈推荐 |
| Qwen2.5 (小) | `ollama pull qwen2.5:7b` | 7B | 8GB | ⭐⭐⭐⭐ | 低配置机器 |
| ChatGLM3 | `ollama pull chatglm3:6b` | 6B | 8GB | ⭐⭐⭐⭐ | 对话友好 |
| DeepSeek-V2 | `ollama pull deepseek-v2:16b` | 16B | 20GB | ⭐⭐⭐⭐⭐ | 推理能力强 |

### 下载 Embeddings 模型（用于向量检索）
```bash
ollama pull nomic-embed-text
```

## 3. 配置 TrendRadar

### 方法 1：环境变量（推荐）

创建 `.env` 文件：
```bash
# LLM 配置
LANGCHAIN_PROVIDER=ollama
LANGCHAIN_MODEL=qwen2.5:14b
LANGCHAIN_BASE_URL=http://localhost:11434  # Ollama 默认地址

# Embeddings 配置（用于向量检索）
LANGCHAIN_EMBEDDINGS_PROVIDER=ollama
LANGCHAIN_EMBEDDINGS_MODEL=nomic-embed-text
LANGCHAIN_EMBEDDINGS_BASE_URL=http://localhost:11434

# 其他配置（可选）
LANGCHAIN_TEMPERATURE=0.3
LANGCHAIN_MAX_TOKENS=2000
LANGCHAIN_TIMEOUT=120
```

### 方法 2：命令行导出
```bash
export LANGCHAIN_PROVIDER=ollama
export LANGCHAIN_MODEL=qwen2.5:14b
export LANGCHAIN_EMBEDDINGS_PROVIDER=ollama
export LANGCHAIN_EMBEDDINGS_MODEL=nomic-embed-text
```

## 4. 运行测试

```bash
# 安装依赖
uv sync --group langchain

# 运行测试
uv run python test_langchain_integration.py
```

## 5. 模型推荐

### 新闻摘要和分析（推荐 Qwen2.5）
```bash
# 高性能（需要 16GB+ 内存）
export LANGCHAIN_MODEL=qwen2.5:14b

# 标准性能（需要 8GB+ 内存）
export LANGCHAIN_MODEL=qwen2.5:7b

# 低配置（需要 4GB+ 内存）
export LANGCHAIN_MODEL=qwen2.5:3b
```

### 对话式问答（推荐 ChatGLM3）
```bash
export LANGCHAIN_MODEL=chatglm3:6b
```

### 趋势预测（推荐 DeepSeek-V2）
```bash
export LANGCHAIN_MODEL=deepseek-v2:16b
export LANGCHAIN_TEMPERATURE=0.5  # 更高创造性
```

## 6. 性能优化

### GPU 加速
Ollama 自动检测并使用 GPU：
- NVIDIA GPU：自动使用 CUDA
- Apple Silicon：自动使用 Metal
- AMD GPU：自动使用 ROCm

### 并发配置
```bash
# Ollama 配置文件 (~/.ollama/config.json)
{
  "num_parallel": 4,  # 并发请求数
  "num_ctx": 4096     # 上下文长度
}
```

## 7. 常见问题

### Q: Ollama 需要网络吗？
A: 只有下载模型时需要网络，运行时完全离线。

### Q: 如何切换模型？
A: 修改环境变量 `LANGCHAIN_MODEL` 即可，无需重启 Ollama。

### Q: 内存不足怎么办？
A: 使用更小的模型，如 `qwen2.5:7b` 或 `qwen2.5:3b`。

### Q: 如何查看已安装的模型？
```bash
ollama list
```

### Q: 如何删除不用的模型？
```bash
ollama rm qwen2.5:14b
```

### Q: Ollama 服务没有启动？
```bash
# 启动 Ollama 服务
ollama serve

# 或在后台运行
ollama serve &
```

## 8. 与 OpenAI 对比

| 维度 | OpenAI (gpt-4o-mini) | Ollama (Qwen2.5-14B) |
|------|----------------------|----------------------|
| **中文能力** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **成本** | $0.15/1M tokens | **免费** |
| **速度** | 快（网络延迟） | **很快（本地）** |
| **隐私** | 数据上传 | **完全本地** |
| **硬件要求** | 无 | 16GB+ RAM |
| **网络依赖** | 必需 | **仅下载时** |
| **稳定性** | API限流 | **无限制** |

## 9. 推荐配置

### 开发环境（本地调试）
```bash
export LANGCHAIN_PROVIDER=ollama
export LANGCHAIN_MODEL=qwen2.5:7b
export LANGCHAIN_EMBEDDINGS_PROVIDER=ollama
export LANGCHAIN_EMBEDDINGS_MODEL=nomic-embed-text
```

### 生产环境（高性能）
```bash
export LANGCHAIN_PROVIDER=ollama
export LANGCHAIN_MODEL=qwen2.5:14b
export LANGCHAIN_EMBEDDINGS_PROVIDER=ollama
export LANGCHAIN_EMBEDDINGS_MODEL=nomic-embed-text
export LANGCHAIN_MAX_TOKENS=4000
```

### 低配置机器
```bash
export LANGCHAIN_PROVIDER=ollama
export LANGCHAIN_MODEL=qwen2.5:3b
export LANGCHAIN_EMBEDDINGS_PROVIDER=ollama
export LANGCHAIN_EMBEDDINGS_MODEL=nomic-embed-text
```

## 10. 进阶使用

### 自定义模型参数
```bash
# 创建自定义 Modelfile
cat > Modelfile <<EOF
FROM qwen2.5:14b

# 设置温度
PARAMETER temperature 0.5

# 设置系统提示
SYSTEM 你是一位专业的新闻分析助手
EOF

# 创建自定义模型
ollama create my-qwen2.5 -f Modelfile

# 使用自定义模型
export LANGCHAIN_MODEL=my-qwen2.5
```

### 远程 Ollama 服务器
```bash
# 如果 Ollama 运行在其他机器上
export LANGCHAIN_BASE_URL=http://192.168.1.100:11434
export LANGCHAIN_EMBEDDINGS_BASE_URL=http://192.168.1.100:11434
```

## 11. 更多资源

- **Ollama 官网**: https://ollama.com
- **模型库**: https://ollama.com/library
- **Qwen2.5 介绍**: https://huggingface.co/Qwen
- **TrendRadar 文档**: ./README.md

---

**快速开始命令**：
```bash
# 1. 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. 下载模型
ollama pull qwen2.5:14b
ollama pull nomic-embed-text

# 3. 配置环境
export LANGCHAIN_PROVIDER=ollama
export LANGCHAIN_MODEL=qwen2.5:14b
export LANGCHAIN_EMBEDDINGS_PROVIDER=ollama

# 4. 运行测试
uv run python test_langchain_integration.py
```

祝你使用愉快！🎉
