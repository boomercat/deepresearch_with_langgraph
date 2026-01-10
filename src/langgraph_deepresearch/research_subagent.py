import os
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel, Field
from typing_extensions import Literal

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, filter_messages
from langchain.chat_models import init_chat_model

from langgraph_deepresearch.state import ResearcherState, ResearcherOutputState
from langgraph_deepresearch.tools import tavily_search, think_tool
from langgraph_deepresearch.utils import get_today_str
from langgraph_deepresearch.prompts import research_agent_prompt_fused , compress_research_system_prompt, compress_research_human_message
from langgraph_deepresearch.tools import get_mcp_client

_client = None

tools = [tavily_search, think_tool]
tools_by_name = {tool.name: tool for tool in tools}

model = init_chat_model(
    model=os.getenv("MODEL_NAME"),          # qwen-max / qwen2.5 / deepseek-r1
    model_provider="openai",               
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api-inference.modelscope.cn/v1/",
    temperature=0.0,
)
model_with_tools = model.bind_tools(tools)

summarization_model = init_chat_model(
    model=os.getenv("SUMMARY_MODEL_NAME"),          # qwen-max / qwen2.5 / deepseek-r1
    model_provider="openai",                
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api-inference.modelscope.cn/v1/",
    temperature=0.0,
)

compress_model =  init_chat_model(
    model=os.getenv("COMPRESS_MODEL_NAME"),          # qwen-max / qwen2.5 / deepseek-r1
    model_provider="openai",                # 关键：强制走 OpenAI-compatible
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api-inference.modelscope.cn/v1/",
    temperature=0.0,
)


async def llm_call(state: ResearcherState):
    """
    基于当前 state 做下一步决策（要不要调用工具 / 直接回答）。

    模型会分析当前 researcher_messages（对话历史 + 已有工具结果），决定：
    1）继续调用搜索/思考等工具以补全信息
    2）信息已经足够，直接给出最终回答（不再 tool_calls）

    返回：
        对 state 的增量更新：追加一条新的 AIMessage 到 researcher_messages。
    """
    client = get_mcp_client()
    mcp_tools = await client.get_tools()
    tools = mcp_tools + [think_tool, tavily_search]
    model_with_tools = model.bind_tools(tools)
    return {
        "researcher_messages": [
            model_with_tools.invoke(
                [SystemMessage(content=research_agent_prompt_fused)] + state["researcher_messages"]
            )
        ]
    }



async def tool_node(state: ResearcherState):
    """
    执行上一条 LLM 输出里请求的所有工具调用（tool_calls）。

    - 从 state["researcher_messages"] 的最后一条消息里取 tool_calls
    - 逐个找到对应的工具函数并执行
    - 把每次工具执行的结果封装成 ToolMessage 写回 messages

    返回：
        对 state 的增量更新：添加 ToolMessage 列表（这些会被 add_messages 机制追加到历史中）
    """
    tool_calls = state["researcher_messages"][-1].tool_calls

    async def execute_tools():
        """
        实际执行所有工具调用的内部协程。

        说明：
        - 每次执行时都会从 MCP server 获取最新的工具列表
        - MCP 工具是异步的，需要使用 ainvoke
        - think_tool 是本地同步工具，需要特殊处理
        """
        # 连接 MCP server，获取最新的工具定义
        client = get_mcp_client()
        mcp_tools = await client.get_tools()

        # MCP 工具 + 本地 think_tool/tavily_search 组合成完整工具列表
        tools = mcp_tools + [think_tool, tavily_search]
        tools_by_name = {tool.name: tool for tool in tools}

        observations = []
        for tool_call in tool_calls:
            tool = tools_by_name[tool_call["name"]]
            if tool_call["name"] == "think_tool":
                # think_tool is sync, use regular invoke
                observation = tool.invoke(tool_call["args"])
            else:
                # MCP tools are async, use ainvoke
                observation = await tool.ainvoke(tool_call["args"])
            observations.append(observation)

        tool_outputs = [
            ToolMessage(
                content=observation,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
            for observation, tool_call in zip(observations, tool_calls)
        ]

        return tool_outputs
    messages = await execute_tools()
    return {"researcher_messages": messages}


def compress_research(state: ResearcherState) -> dict:
    """
    把研究过程中收集到的内容“压缩成一段简洁总结”，并整理 raw notes。

    用途：
    - research loop 结束后，给 supervisor/上层节点一个可读的结论
    - 同时保留原始工具输出和关键中间推理（raw_notes）便于溯源

    返回：
        {
          "compressed_research": "...压缩后的总结...",
          "raw_notes": ["...拼接后的原始记录..."]
        }
    """
    system_message = compress_research_system_prompt.format(date=get_today_str())

    # 压缩时的输入通常包括：
    # - System：告诉模型以“压缩总结”的角色工作
    # - researcher_messages：完整过程（含 ToolMessage）
    # - Human：提示“请你压缩整理”
    messages = (
        [SystemMessage(content=system_message)]
        + state.get("researcher_messages", [])
        + [HumanMessage(content=compress_research_human_message)]
    )

    response = compress_model.invoke(messages)

    # 从历史消息里抽取“工具输出 + AI 输出”作为原始笔记 raw_notes
    # include_types=["tool","ai"] 表示只抓工具结果与模型总结（不抓 human/system）
    raw_notes = [
        str(m.content)
        for m in filter_messages(
            state["researcher_messages"],
            include_types=["tool", "ai"]
        )
    ]

    return {
        "compressed_research": str(response.content),
        "raw_notes": ["\n".join(raw_notes)]
    }



def should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    """
    判断下一步走向：继续调用工具，还是结束并压缩输出。

    - 如果最后一条 AIMessage 里包含 tool_calls：说明模型要用工具 → 去 tool_node
    - 否则：说明模型已经准备好最终答案（不再调用工具） → 去 compress_research

    返回：
        "tool_node" 或 "compress_research"
    """
    messages = state["researcher_messages"]
    last_message = messages[-1]

    # 如果最后一条消息有工具调用请求，就继续走工具节点
    if last_message.tool_calls:
        return "tool_node"

    # 否则说明已经产出最终答复，不需要工具了，进入压缩总结节点
    return "compress_research"



# 构建状态机：输入状态是 ResearcherState，输出结构是 ResearcherOutputState
agent_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("compress_research", compress_research)


agent_builder.add_edge(START, "llm_call")

# llm_call 后根据 should_continue 的返回值做分支跳转：
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node",             
        "compress_research": "compress_research" 
    },
)

agent_builder.add_edge("tool_node", "llm_call")

agent_builder.add_edge("compress_research", END)

researcher_agent = agent_builder.compile()

async def main():

    research_brief = """我希望识别并评估东北地区范围内、以咖啡品质著称的咖啡店。研究应聚焦于对东北地区咖啡店的分析与比较，并以咖啡品质作为唯一的核心评判标准。

    在咖啡品质的评估方法上不作限制，例如可以参考专家评测、顾客评分、精品咖啡认证等多种方式。除非某些因素会直接影响人们对咖啡品质的感知，否则不考虑咖啡店的氛围、地理位置、是否提供 Wi-Fi 或餐食等因素。

    研究过程中请优先参考一手与权威来源，包括但不限于：咖啡店的官方网站、可信的第三方咖啡评测机构，以及 百度、美团 等主流评价平台中直接反映顾客对咖啡品质评价的内容。

    研究最终应形成一份有充分依据支撑的东北地区顶级咖啡店清单或排名，重点突出各咖啡店在咖啡品质方面的表现。"""

    result = await researcher_agent.ainvoke({"researcher_messages": [HumanMessage(content=f"{research_brief}.")]})
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())