"""
LangChain 1.0 集成测试脚本

测试 NewsAnalystAgent 的基本功能
"""

import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """测试导入"""
    print("=" * 60)
    print("测试 1: 检查导入")
    print("=" * 60)

    try:
        from langchain_agents.config import get_config, LangChainConfig

        print("✅ langchain_agents.config 导入成功")

        from langchain_agents.tools.trendradar_tools import (
            GetLatestNewsTool,
            AnalyzeTrendTool,
            SearchNewsTool,
        )

        print("✅ langchain_agents.tools 导入成功")

        from langchain_agents.agents.news_analyst import NewsAnalystAgent

        print("✅ langchain_agents.agents 导入成功")

        print("\n🎉 所有模块导入成功！\n")
        return True

    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_config():
    """测试配置加载"""
    print("=" * 60)
    print("测试 2: 配置管理")
    print("=" * 60)

    try:
        from langchain_agents.config import get_config

        # 检查是否设置了 API Key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print(
                "⚠️  警告: OPENAI_API_KEY 未设置 (这会导致 Agent 创建失败)"
            )
            print("   设置方式: export OPENAI_API_KEY=sk-xxx")
            return False

        config = get_config()
        print(f"✅ 配置加载成功")
        print(f"   - LLM Provider: {config.llm.provider}")
        print(f"   - Model: {config.llm.model}")
        print(f"   - Temperature: {config.llm.temperature}")
        print(f"   - Max Tokens: {config.llm.max_tokens}")
        print(f"   - Timeout: {config.llm.timeout}s")
        print(f"   - Vector Store: {config.vector_store.provider}")

        return True

    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_tools():
    """测试工具包装器"""
    print("\n" + "=" * 60)
    print("测试 3: TrendRadar Tools 包装器")
    print("=" * 60)

    try:
        from langchain_agents.tools.trendradar_tools import (
            get_all_trendradar_tools,
        )

        tools = get_all_trendradar_tools()
        print(f"✅ 获取工具成功，共 {len(tools)} 个工具:")

        for tool in tools:
            print(f"   - {tool.name}: {tool.description[:50]}...")

        return True

    except Exception as e:
        print(f"❌ 工具加载失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_agent_creation():
    """测试 Agent 创建"""
    print("\n" + "=" * 60)
    print("测试 4: NewsAnalystAgent 创建")
    print("=" * 60)

    # 检查 API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  跳过: OPENAI_API_KEY 未设置")
        return False

    try:
        from langchain_agents.agents.news_analyst import NewsAnalystAgent

        print("正在创建 NewsAnalystAgent...")
        agent = NewsAnalystAgent()

        print("✅ Agent 创建成功")
        print(f"   - LLM: {agent.llm}")
        print(f"   - Tools: {len(agent.tools)} 个")
        print(f"   - System Prompt: {agent.system_prompt[:50]}...")

        return True

    except Exception as e:
        print(f"❌ Agent 创建失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_version_info():
    """测试版本信息"""
    print("\n" + "=" * 60)
    print("测试 5: LangChain 版本信息")
    print("=" * 60)

    try:
        import langchain
        import langchain_core
        import langchain_openai
        import langchain_community

        print(f"✅ LangChain 版本:")
        print(f"   - langchain: {langchain.__version__}")
        print(f"   - langchain-core: {langchain_core.__version__}")

        # langchain_openai 可能没有 __version__ 属性
        try:
            print(f"   - langchain-openai: {langchain_openai.__version__}")
        except AttributeError:
            print(f"   - langchain-openai: (已安装)")

        try:
            print(f"   - langchain-community: {langchain_community.__version__}")
        except AttributeError:
            print(f"   - langchain-community: (已安装)")

        # 检查是否是 1.0.x
        if langchain.__version__.startswith("1.0"):
            print("\n🎉 成功升级到 LangChain 1.0.x!")
        else:
            print(
                f"\n⚠️  警告: 当前版本是 {langchain.__version__}, 不是 1.0.x"
            )

        return True

    except Exception as e:
        print(f"❌ 版本检查失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n🧪 TrendRadar LangChain 1.0 集成测试\n")

    tests = [
        ("导入测试", test_imports),
        ("配置测试", test_config),
        ("工具测试", test_tools),
        ("Agent 创建测试", test_agent_creation),
        ("版本信息", test_version_info),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} 异常: {e}")
            results.append((name, False))

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {name}")

    print(f"\n总计: {passed}/{total} 通过")

    if passed == total:
        print("\n🎉 所有测试通过! LangChain 1.0 集成成功!")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查上述错误信息")
        return 1


if __name__ == "__main__":
    sys.exit(main())
