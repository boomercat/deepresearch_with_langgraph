import os
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from datetime import datetime
from typing_extensions import Annotated, List, Literal

from langchain.chat_models import init_chat_model 
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, InjectedToolArg
from tavily import TavilyClient
from langgraph_deepresearch.utils import get_today_str
from langgraph_deepresearch.state import Summary
from langgraph_deepresearch.prompts import summarize_webpage_prompt






summarization_model =  init_chat_model(
    model=os.getenv("SUMMARY_MODEL_NAME"),          # qwen-max / qwen2.5 / deepseek-r1
    model_provider="openai",                # 关键：强制走 OpenAI-compatible
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api-inference.modelscope.cn/v1/",
    temperature=0.0,
)

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY"),
)

#----------搜索函数---------
def tavily_search_multiple(
    search_queries: List[str],
    max_results: int = 3,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
) -> List[dict]:
    """
    使用 Tavily API 批量执行多个搜索 query。

    参数:
        search_queries: 要执行的搜索查询列表（每个元素是一条 query）
        max_results: 每条 query 最多返回多少条结果
        topic: 搜索的主题过滤（可选：general/news/finance）
        include_raw_content: 是否把网页原始内容一起返回（用于后续摘要）

    返回:
        一个列表，每个元素对应一条 query 的 Tavily 返回结果（dict）
    """

    # 这里是“串行”执行搜索：一个 query 搜完再搜下一个
    # 备注：如果你希望并行加速，可以用 AsyncTavilyClient
    search_docs = []
    for query in search_queries:
        result = tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic
        )
        search_docs.append(result)

    return search_docs


def summarize_webpage_content(webpage_content: str) -> str:
    """
    使用配置好的 summarization_model 对网页原文做摘要（结构化输出）。

    参数:
        webpage_content: 网页原始内容（raw content），通常比较长

    返回:
        格式化后的摘要字符串：
        - <summary>...</summary>
        - <key_excerpts>...</key_excerpts>

    失败兜底:
        如果摘要失败，则返回网页前 1000 字符（避免直接崩掉）
    """
    try:
        # 让 summarization_model 以结构化方式输出 Summary schema（避免自然语言不好解析）
        structured_model = summarization_model.with_structured_output(
            Summary,
            method="function_calling"   # 👈 关键修改
        )

        # 调用模型生成摘要
        summary = structured_model.invoke([
            HumanMessage(content=summarize_webpage_prompt.format(
                webpage_content=webpage_content,
                date=get_today_str()
            ))
        ])

        # 输出成统一的可读格式（方便后续拼接给 research agent）
        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )

        return formatted_summary

    except Exception as e:
        # 任何异常都兜底，避免搜索工具整体失败
        print(f"网页摘要失败: {str(e)}")
        return webpage_content[:1000] + "..." if len(webpage_content) > 1000 else webpage_content

def deduplicate_search_results(search_results: List[dict]) -> dict:
    """
    根据 URL 对搜索结果去重，避免同一个网页被重复处理（浪费 token + 时间）。

    参数:
        search_results: tavily_search_multiple 返回的结果列表（每个 query 一个 dict）

    返回:
        dict: {url -> result_dict} 的映射，只保留每个 URL 的第一条出现结果
    """
    unique_results = {}

    for response in search_results:
        for result in response["results"]:
            url = result["url"]
            if url not in unique_results:
                unique_results[url] = result

    return unique_results


def process_search_results(unique_results: dict) -> dict:
    """
    对去重后的结果做进一步处理：如果有 raw_content 就做摘要，否则用短 content 兜底。

    参数:
        unique_results: deduplicate_search_results 输出的 {url -> result} 字典

    返回:
        summarized_results: {url -> {"title": ..., "content": ...}}，
        其中 content 已经是“可读摘要”或“短内容”
    """
    summarized_results = {}

    for url, result in unique_results.items():
        # 如果没有 raw_content，就只能用 Tavily 自带的 content（一般比较短）
        if not result.get("raw_content"):
            content = result["content"]
        else:
            # 有 raw_content 时，优先对原文做摘要，提升质量并减少上下文长度
            content = summarize_webpage_content(result["raw_content"])

        summarized_results[url] = {
            "title": result["title"],
            "content": content
        }

    return summarized_results



def format_search_output(summarized_results: dict) -> str:
    """
    把处理后的搜索结果整理成统一的字符串输出（带 SOURCE 分隔）。

    参数:
        summarized_results: process_search_results 的输出 {url -> {"title","content"}}

    返回:
        一个格式化字符串，形如：
        Search results:
        --- SOURCE 1: title ---
        URL: ...
        SUMMARY: ...
        --------------------------------------------------------------------------------
    """
    if not summarized_results:
        return "没有找到有效搜索结果，请尝试换 query 或更换搜索 API。"

    formatted_output = "Search results:\n\n"

    for i, (url, result) in enumerate(summarized_results.items(), 1):
        formatted_output += f"\n\n--- SOURCE {i}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "-" * 80 + "\n"

    return formatted_output


@tool(parse_docstring=True)
def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 3,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> str:

    """Fetch results from Tavily search API with content summarization.

    Args:
        query: A single search query to execute
        max_results: Maximum number of results to return
        topic: Topic to filter results by ('general', 'news', 'finance')

    Formatted string of search results with summaries
    """

    # 这里内部复用 tavily_search_multiple：把单 query 转成 list，统一处理流程
    search_results = tavily_search_multiple(
        [query],
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
    )

    # 1) 先按 URL 去重
    unique_results = deduplicate_search_results(search_results)

    # 2) 对每个网页做摘要（如果有 raw_content）
    summarized_results = process_search_results(unique_results)

    # 3) 统一格式化输出
    return format_search_output(summarized_results)

    # 
    # 思考工具：让 Agent 在每次搜索后做“战略性复盘”，避免无脑继续搜。

    # 建议使用时机：
    # - 拿到搜索结果后：我找到了哪些关键事实？
    # - 决定下一步前：是否已足够回答？还是需要继续搜索？
    # - 发现缺口时：还缺哪些关键信息？下一步应该搜什么？
    # - 结束前：证据是否充分？能否组织成高质量回答？

    # reflection 建议包含四点：
    # 1) 当前结论：我拿到了哪些具体信息？
    # 2) 缺口分析：还缺哪些关键点？
    # 3) 质量评估：证据/例子是否足够？
    # 4) 下一步决策：继续搜？还是直接写答案？

    # 参数:
    #     reflection: 复盘内容（Agent 自己写的）

    # 返回:
    #     确认信息（告诉 Agent 已记录复盘）
    # """
@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:

    """Tool for strategic reflection on research progress and decision-making.
    
    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.
    
    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?
    
    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?
    
    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps
        
    Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"