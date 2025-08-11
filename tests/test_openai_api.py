"""
测试 OpenAI API 兼容服务的客户端
"""

import asyncio
import datetime
import io

import aiohttp

# 服务器配置
BASE_URL = "http://localhost:8008"

# 全局会话对象
_session = None


async def get_session():
    """获取或创建全局会话对象"""
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession()
    return _session


async def close_session():
    """关闭全局会话对象"""
    global _session
    if _session and not _session.closed:
        await _session.close()
        _session = None


async def test_models_endpoint():
    """测试模型列表接口"""
    print("🔍 测试 /v1/models 接口...")
    session = await get_session()
    async with session.get(f"{BASE_URL}/v1/models") as response:
        if response.status == 200:
            data = await response.json()
            print(f"✅ 成功获取模型列表，共 {len(data['data'])} 个模型")
            for model in data["data"][:]:  # 只显示前3个
                print(f"   - {model['id']} (owned by {model['owned_by']})")
            return True
        else:
            print(f"❌ 请求失败: {response.status}")
            return False


async def test_chat_completion():
    """测试聊天完成接口"""
    start_time = datetime.datetime.now()
    print(f"\n💬 测试 /v1/chat/completions 接口... [开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}]")

    # payload = {
    #     "model": "qwen3_235b_a22b_instruct_2507_fp8",
    #     "messages": [
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": "Hello! Can you explain what Python is?"}
    #     ],
    #     "max_tokens": 150,
    #     "temperature": 0.7
    # }

    system = """
# Role: 任务分类器

## Profile
- description: 专业高效的任务分类AI，准确分类用户问题到指定类别
- background: 工业和企业应用场景的智能分类系统
- personality: 简洁高效
- expertise: 任务分类

## Skills
1. 分类识别
   - 语义分析
   - 关键词提取
   - 上下文关联

2. 工业知识
   - 产品识别
   - 工艺理解

## Rules
1. 分类原则：
   - 严格遵循四分类标准: 中控、工艺、事实、实时查询
   - 仅返回分类结果单词
   - 保持一致性

2. 行为准则：
   - 绝对中立
   - 最小化输出
   - 拒绝解释

3. 限制条件：
   - 不处理模糊分类
   - 不扩展功能

## Workflows
- 接收用户输入
- 分析语义和关键词
- 匹配分类标准
- 返回分类结果

## Initialization
作为任务分类器，严格遵守Rules执行分类任务。
    """
    payload = {
        # "model": "qwen3_0.6b",
        "model": "qwen3_235b_a22b_instruct_2507_fp8",
        "messages": [
            {"role": "system", "content": system},
            # {"role": "user", "content": "聚丙烯有哪些工艺 \\no_think"}
            # {"role": "user", "content": "中控有哪些产品 \\o_think"}
            {"role": "user", "content": "介绍下 OMC"},
            # {"role": "user", "content": "太阳距离地球有多远"},
        ],
        "max_tokens": 150,
        "temperature": 0.0,
    }

    session = await get_session()
    async with session.post(f"{BASE_URL}/v1/chat/completions", json=payload, headers={"Content-Type": "application/json"}) as response:
        if response.status == 200:
            data = await response.json()
            print(f"data: {data}")
            print("✅ 聊天完成成功")
            print(f"   模型: {data['model']}")
            print(f"   响应: {data['choices'][0]['message']['content'][:100]}")
            print(f"   Token使用: {data['usage']['total_tokens']}")

            end_time = datetime.datetime.now()
            elapsed_time = end_time - start_time
            elapsed_ms = elapsed_time.total_seconds() * 1000
            print(f"   耗时: {elapsed_ms:.2f} 毫秒 [结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}]")

            return True
        else:
            print(f"❌ 请求失败: {response.status}")
            error = await response.text()
            print(f"   错误信息: {error}")
            return False


async def test_chat_completion2():
    """测试聊天完成接口"""
    start_time = datetime.datetime.now()
    print(f"\n💬 测试 /v1/chat/completions 接口... [开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}]")

    # payload = {
    #     "model": "qwen3_235b_a22b_instruct_2507_fp8",
    #     "messages": [
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": "Hello! Can you explain what Python is?"}
    #     ],
    #     "max_tokens": 150,
    #     "temperature": 0.7
    # }

    system = """
你是个任务分类器，任务的种类有：
- 中控
- 工艺
- 事实
- 实时查询

你对用户的回答只需返回上述类别即可，无需回答任何其他文本。其他的信息有：
中控包括有 OMC，Nyx 等产品。

例如：
问题1：
用户问题：中控有哪些产品
回答：中控
问题2：
用户问题：今天杭州天气怎么样？
回答：实时查询
问题3：
用户问题：聚丙烯都有哪些流程？
回答：工艺
问题4：
用户问题：人有几条腿？
回答：事实
    """
    payload = {
        # "model": "qwen3_0.6b",
        "model": "qwen3_235b_a22b_instruct_2507_fp8",
        "messages": [
            {"role": "system", "content": system},
            # {"role": "user", "content": "聚丙烯有哪些工艺"},
            # {"role": "user", "content": "中控有哪些产品"},
            # {"role": "user", "content": "介绍下 OMC"},
            # {"role": "user", "content": "太阳距离地球有多远"},
            {"role": "user", "content": "杭州今天天气怎么样"},
        ],
        "max_tokens": 150,
        "temperature": 0.0,
    }

    session = await get_session()
    async with session.post(f"{BASE_URL}/v1/chat/completions", json=payload, headers={"Content-Type": "application/json"}) as response:
        if response.status == 200:
            data = await response.json()
            print(f"data: {data}")
            print("✅ 聊天完成成功")
            print(f"   模型: {data['model']}")
            print(f"   响应: {data['choices'][0]['message']['content'][:100]}")
            print(f"   Token使用: {data['usage']['total_tokens']}")

            end_time = datetime.datetime.now()
            elapsed_time = end_time - start_time
            elapsed_ms = elapsed_time.total_seconds() * 1000
            print(f"   耗时: {elapsed_ms:.2f} 毫秒 [结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}]")

            return True
        else:
            print(f"❌ 请求失败: {response.status}")
            error = await response.text()
            print(f"   错误信息: {error}")
            return False


async def test_embeddings(model="jina-embeddings-v3"):
    """测试嵌入接口"""
    # qwen3_embedding_4b
    print("\n🔢 测试 /v1/embeddings 接口...")

    payload = {"model": model, "input": ["Hello world", "Python programming", "Machine learning"], "task": "retrieval.passage"}

    session = await get_session()
    async with session.post(f"{BASE_URL}/v1/embeddings", json=payload, headers={"Content-Type": "application/json"}) as response:
        if response.status == 200:
            data = await response.json()
            print("✅ 嵌入生成成功")
            print(f"   模型: {data['model']}")
            print(f"   嵌入数量: {len(data['data'])}")
            print(f"   嵌入维度: {len(data['data'][0]['embedding'])}")
            print(f"   Token使用: {data['usage']['total_tokens']}")
            return True
        else:
            print(f"❌ 请求失败: {response.status}")
            return False


async def test_rerank():
    """测试重排序接口"""
    print("\n📊 测试 /v1/rerank 接口...")

    payload = {
        "model": "jina-reranker",
        "query": "machine learning",
        "documents": ["Python is a programming language", "Machine learning is a subset of artificial intelligence", "Deep learning uses neural networks", "Data science involves analyzing data"],
        "top_k": 3,
    }

    session = await get_session()
    async with session.post(f"{BASE_URL}/v1/rerank", json=payload, headers={"Content-Type": "application/json"}) as response:
        if response.status == 200:
            data = await response.json()
            print("✅ 重排序成功")
            print(f"   模型: {data['model']}")
            print(f"   结果数量: {len(data['data'])}")
            for i, result in enumerate(data["data"]):
                print(f"   {i + 1}. 分数: {result['relevance_score']:.3f} - {result['document'][:50]}...")
            return True
        else:
            print(f"❌ 请求失败: {response.status}")
            return False


async def test_tts():
    """测试文本转语音接口"""
    print("\n🔊 测试 /v1/audio/speech 接口...")

    payload = {"model": "tts-1", "input": "Hello, this is a test of text to speech functionality.", "voice": "alloy", "response_format": "mp3"}

    session = await get_session()
    async with session.post(f"{BASE_URL}/v1/audio/speech", json=payload, headers={"Content-Type": "application/json"}) as response:
        if response.status == 200:
            audio_data = await response.read()
            print("✅ TTS 生成成功")
            print(f"   音频数据大小: {len(audio_data)} 字节")

            # 可选：保存音频文件
            with open("test_output.mp3", "wb") as f:
                f.write(audio_data)
            print("   音频已保存为 test_output.mp3")
            return True
        else:
            print(f"❌ 请求失败: {response.status}")
            return False


async def test_asr():
    """测试语音转文本接口"""
    print("\n🎤 测试 /v1/audio/transcriptions 接口...")

    # 创建一个模拟的音频文件
    fake_audio_data = b"fake_audio_content_for_testing"

    data = aiohttp.FormData()
    data.add_field("file", io.BytesIO(fake_audio_data), filename="test_audio.mp3", content_type="audio/mpeg")
    data.add_field("model", "whisper-1")
    data.add_field("response_format", "json")

    session = await get_session()
    async with session.post(f"{BASE_URL}/v1/audio/transcriptions", data=data) as response:
        if response.status == 200:
            data = await response.json()
            print("✅ ASR 转录成功")
            print(f"   转录结果: {data['text']}")
            return True
        else:
            print(f"❌ 请求失败: {response.status}")
            error = await response.text()
            print(f"   错误信息: {error}")
            return False


async def test_deep_search():
    """测试深度搜索接口"""
    print("\n🔍 测试 /v1/deep-search 接口...")

    payload = {
        "model": "jina-deepsearch",
        "messages": [{"role": "user", "content": "What are the latest developments in artificial intelligence?"}],
        "max_tokens": 300,
        # "search_depth": 3,
        # "include_sources": True
    }

    session = await get_session()
    async with session.post(f"{BASE_URL}/v1/deep-search", json=payload, headers={"Content-Type": "application/json"}) as response:
        if response.status == 200:
            data = await response.json()
            print("✅ 深度搜索成功")
            print(f"   模型: {data['model']}")

            # 获取响应内容
            content = data["choices"][0]["message"]["content"]
            print(f"   响应长度: {len(content)} 字符")

            # 显示响应的前200个字符
            print(f"   响应预览: {content[:200]}...")

            # 如果有usage信息，显示token使用情况
            if "usage" in data:
                usage = data["usage"]
                print(f"   Token使用: {usage.get('total_tokens', 'N/A')}")
                if "prompt_tokens" in usage:
                    print(f"   输入Token: {usage['prompt_tokens']}")
                if "completion_tokens" in usage:
                    print(f"   输出Token: {usage['completion_tokens']}")

            # 检查是否包含引用信息
            if "citations" in content or "url" in content.lower():
                print("   ✅ 包含引用信息")

            return True
        else:
            print(f"❌ 请求失败: {response.status}")
            error = await response.text()
            print(f"   错误信息: {error}")
            return False


async def test_classify():
    """测试文本分类接口"""
    start_time = datetime.datetime.now()
    print(f"\n🏷️ 测试 /v1/classify 接口... [开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}]")

    payload = {
        "model": "jina-classify",
        "input": [
            "中控有哪些产品",
            # "聚丙烯都有哪些流程",
            # "今天杭州天气怎么样",
        ],
        "labels": ["中控", "工艺", "日常"],
    }

    session = await get_session()
    async with session.post(f"{BASE_URL}/v1/classify", json=payload, headers={"Content-Type": "application/json"}) as response:
        if response.status == 200:
            data = await response.json()
            print("✅ 文本分类成功")
            print(f"   模型: {data['model']}")
            print(f"   结果: {len(data['data'])} 个文本分类")

            # 显示每个文本的分类结果
            for i, text_results in enumerate(data["data"]):
                print(f"   文本 {i + 1}: {len(text_results)} 个标签的分数")

                # 按分数排序显示所有标签
                sorted_results = sorted(text_results, key=lambda x: x["score"], reverse=True)
                for j, result in enumerate(sorted_results):
                    indicator = "★" if j == 0 else " "  # 最高分数的标签用星号标记
                    print(f"      {indicator} {result['label']}: {result['score']:.4f}")
                print()  # 空行分隔

            if "usage" in data:
                print(f"   Token使用: {data['usage']['total_tokens']}")

            # 计算并显示耗时
            end_time = datetime.datetime.now()
            elapsed_time = end_time - start_time
            elapsed_ms = elapsed_time.total_seconds() * 1000
            print(f"   耗时: {elapsed_ms:.2f} 毫秒 [结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}]")

            return True
        else:
            print(f"❌ 请求失败: {response.status}")
            error = await response.text()
            print(f"   错误信息: {error}")
            return False


async def test_search():
    """测试搜索接口"""
    print("\n🔍 测试 /v1/search 接口...")
    session = await get_session()
    payload = {"model": "jina-search", "query": "Jina AI", "top_k": 5}

    async with session.get(f"{BASE_URL}/v1/search", json=payload) as response:
        if response.status == 200:
            data = await response.json()
            print(f"✅ 搜索成功，共返回 {len(data['data'])} 条结果")

            # 打印搜索结果
            for i, result in enumerate(data["data"]):
                print(f"\n结果 #{i + 1}:")
                print(f"标题: {result['title']}")
                print(f"URL: {result['url']}")
                print(f"描述: {result['description'][:100]}..." if len(result["description"]) > 100 else f"描述: {result['description']}")
                if result.get("date"):
                    print(f"日期: {result['date']}")

            return True
        else:
            error_text = await response.text()
            print(f"❌ 搜索请求失败: {response.status}\n{error_text}")
            return False


async def run_all_tests():
    """运行所有测试"""
    print("🚀 开始测试 OpenAI API 兼容服务...")
    print("=" * 50)

    tests = [
        # test_models_endpoint,
        test_chat_completion,
        test_chat_completion2,
        # test_embedding,
        # test_rerank,
        # test_tts,
        # test_asr,
        # test_deep_search,
        # test_classify,
        # test_search
    ]

    results = []
    try:
        for test in tests:
            try:
                result = await test()
                results.append(result)
            except Exception as e:
                print(f"❌ 测试异常: {e}")
                results.append(False)

            await asyncio.sleep(0.5)  # 短暂延迟
    finally:
        # 确保关闭会话
        await close_session()

    print("\n" + "=" * 50)
    print("📋 测试结果汇总:")
    test_names = ["模型列表", "聊天完成", "文本嵌入", "文档重排序", "文本转语音", "语音转文本", "深度搜索", "文本分类"]

    for i, (name, result) in enumerate(zip(test_names, results, strict=False)):
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {i + 1}. {name}: {status}")

    success_count = sum(results)
    total_count = len(results)
    print(f"\n🎯 总体结果: {success_count}/{total_count} 个测试通过")

    if success_count == total_count:
        print("🎉 所有测试都通过了！API 服务运行正常。")
    else:
        print("⚠️  部分测试失败，请检查服务器状态和配置。")


if __name__ == "__main__":
    print("请确保 FastAPI 服务器正在运行 (python fastapi_openai_server.py)")
    print("服务器地址: http://localhost:8000")
    print()

    try:
        asyncio.run(run_all_tests())
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试运行出错: {e}")
    # asyncio.run 会自动关闭事件循环，所以不需要在这里显式关闭会话
