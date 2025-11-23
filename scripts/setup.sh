#!/bin/bash
#
# TrendRadar Configuration Setup
#
# 一键初始化配置系统：
# 1. 从 config.schema.yaml 生成 config/local.yaml
# 2. 从 config/local.yaml 生成 .env
#
# 用法:
#   ./scripts/setup.sh          # 正常模式（保留现有配置）
#   ./scripts/setup.sh --force  # 强制模式（覆盖现有配置）
#

set -e  # 遇到错误立即退出

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

success() {
    echo -e "${GREEN}✓${NC} $1"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

error() {
    echo -e "${RED}✗${NC} $1"
}

# 检查 Python
check_python() {
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is not installed"
        exit 1
    fi

    success "Python 3 found: $(python3 --version)"
}

# 检查 PyYAML
check_pyyaml() {
    if ! python3 -c "import yaml" 2>/dev/null; then
        warning "PyYAML is not installed"
        info "Installing PyYAML..."

        if command -v uv &> /dev/null; then
            # 使用 uv（如果可用）
            uv pip install pyyaml
        elif command -v pip3 &> /dev/null; then
            # 使用 pip3
            pip3 install pyyaml
        else
            error "Neither 'uv' nor 'pip3' found. Please install PyYAML manually:"
            error "  pip install pyyaml"
            exit 1
        fi

        success "PyYAML installed"
    else
        success "PyYAML found"
    fi
}

# 主函数
main() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  TrendRadar Configuration Setup"
    echo "═══════════════════════════════════════════════════════"
    echo ""

    # 切换到项目根目录
    cd "${PROJECT_ROOT}"

    # 检查依赖
    info "Checking dependencies..."
    check_python
    check_pyyaml
    echo ""

    # 步骤 1: 生成 config/local.yaml
    info "Step 1: Generating config/local.yaml from schema..."
    if python3 scripts/lib/generate_from_schema.py "$@"; then
        success "config/local.yaml generated"
    else
        error "Failed to generate config/local.yaml"
        exit 1
    fi
    echo ""

    # 步骤 2: 生成 .env
    info "Step 2: Generating .env from config/local.yaml..."
    if python3 scripts/lib/generate_env.py; then
        success ".env generated"
    else
        error "Failed to generate .env"
        exit 1
    fi
    echo ""

    # 完成
    echo "═══════════════════════════════════════════════════════"
    success "Configuration setup completed!"
    echo "═══════════════════════════════════════════════════════"
    echo ""

    # 提示后续步骤
    echo "📝 Next steps:"
    echo ""
    echo "1. ${YELLOW}Edit configuration${NC} (optional):"
    echo "   vim config/local.yaml"
    echo ""
    echo "2. ${YELLOW}Set API keys${NC} (if using OpenAI/Anthropic):"
    echo "   # Option A: Edit config/local.yaml"
    echo "   vim config/local.yaml  # Find api_keys.openai_api_key"
    echo ""
    echo "   # Option B: Set environment variable directly"
    echo "   export OPENAI_API_KEY=sk-xxx"
    echo ""
    echo "3. ${YELLOW}Regenerate .env${NC} (if you edited local.yaml):"
    echo "   ./scripts/setup.sh"
    echo ""
    echo "4. ${YELLOW}Load environment variables${NC}:"
    echo "   source .env"
    echo ""
    echo "5. ${YELLOW}Run tests${NC}:"
    echo "   uv run python test_langchain_integration.py"
    echo ""

    # Ollama 提示
    if [ -f "config/local.yaml" ]; then
        PROVIDER=$(grep -A1 "^llm:" config/local.yaml | grep "provider:" | awk '{print $2}' | tr -d '"' || echo "")

        if [ "$PROVIDER" = "ollama" ]; then
            echo "📌 ${BLUE}Detected Ollama provider${NC}"
            echo ""
            echo "   Make sure Ollama is installed and running:"
            echo "   ollama serve"
            echo ""
            echo "   Download models:"
            echo "   ollama pull qwen2.5:14b          # LLM"
            echo "   ollama pull nomic-embed-text     # Embeddings"
            echo ""
            echo "   See OLLAMA_SETUP.md for more details"
            echo ""
        fi
    fi
}

# 运行主函数
main "$@"
