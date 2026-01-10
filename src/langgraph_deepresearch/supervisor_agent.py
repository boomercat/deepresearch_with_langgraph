import asyncio
import os
from dotenv import load_dotenv
load_dotenv()
from typing_extensions import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage, 
    BaseMessage, 
    SystemMessage, 
    ToolMessage,
    filter_messages
)
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from langgraph_deepresearch.prompts import lead_researcher_prompt
from langgraph_deepresearch.research_subagent import researcher_agent
from langgraph_deepresearch.state import SupervisorState, ConductResearch, ResearchComplete
from langgraph_deepresearch.utils import  format_messages, get_today_str
from langgraph_deepresearch.tools import think_tool


supervisor_tools = [ConductResearch, ResearchComplete, think_tool]
supervisor_model = init_chat_model(
    model=os.getenv("MODEL_NAME"),          # qwen-max / qwen2.5 / deepseek-r1
    model_provider="openai",                
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api-inference.modelscope.cn/v1/",
    temperature=0.0,
)
supervisor_model_with_tools = supervisor_model.bind_tools(supervisor_tools)

max_researcher_iterations = 6  # think_tool + ConductResearch 的调用次数上限

# Supervisor 允许同时并发启动的最大 Research 子智能体数量
max_concurrent_researchers = 3


def get_notes_from_tool_calls(messages: list[BaseMessage]) -> list[str]:
    """
    从 Supervisor 的消息历史中提取研究笔记。

    该函数用于从 ToolMessage 中提取 Research 子智能体返回的
    「压缩研究结果（compressed research）」。

    在 Supervisor 通过 ConductResearch 工具将研究任务委派给子智能体后，
    每个子智能体都会将其研究结论以 ToolMessage.content 的形式返回。
    本函数会遍历 Supervisor 的消息历史，筛选出所有 ToolMessage，
    并提取其中的内容，最终汇总为研究笔记列表。

    Args:
        messages: Supervisor 会话历史中的消息列表

    Returns:
        从 ToolMessage 中提取出的研究笔记字符串列表
    """
    return [
        tool_msg.content
        for tool_msg in filter_messages(messages, include_types="tool")
    ]


async def supervisor(state: SupervisorState) -> Command[Literal["supervisor_tools"]]:
    """
    Supervisor 主协调节点。

    负责分析研究任务说明（research_brief）以及当前研究进展，
    决定下一步行动，包括：
    - 需要进一步研究的具体主题
    - 是否并行启动多个 Research 子智能体
    - 当前研究流程是否可以结束

    Args:
        state: 当前 Supervisor 的状态，包含消息记录和研究进度信息

    Command：跳转至 supervisor_tools 节点，并携带更新后的状态
    """
    supervisor_messages = state.get("supervisor_messages", [])

    # 构造 system message，注入当前日期与并发 / 迭代约束条件
    system_message = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=max_concurrent_researchers,
        max_researcher_iterations=max_researcher_iterations
    )
    messages = [SystemMessage(content=system_message)] + supervisor_messages

    # 调用带工具能力的 Supervisor 模型，决定下一步研究动作
    response = await supervisor_model_with_tools.ainvoke(messages)

    return Command(
        goto="supervisor_tools",
        update={
            # 记录 Supervisor 最新一次决策消息
            "supervisor_messages": [response],
            # 研究轮次计数 +1
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )


async def supervisor_tools(state: SupervisorState) -> Command[Literal["supervisor", "__end__"]]:
    """
    Supervisor 工具执行节点。

    负责执行 Supervisor 在上一轮中给出的所有工具调用决策，包括：
    - 执行 think_tool，用于策略反思与规划
    - 并行启动多个 Research 子智能体执行具体研究任务
    - 汇总并聚合研究结果
    - 判断研究流程是否结束

    Args:
        state: 当前 Supervisor 状态，包含消息记录与迭代计数

    Returns:
        Command：继续监督流程、结束流程，或在异常情况下终止
    """
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]

    # -------- 初始化统一返回所需的变量 --------
    tool_messages = []
    all_raw_notes = []
    next_step = "supervisor"  # 默认返回 Supervisor 节点
    should_end = False

    # -------- 研究流程终止条件判断 --------
    exceeded_iterations = research_iterations >= max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    if exceeded_iterations or no_tool_calls or research_complete:
        # 满足任一条件即终止研究流程
        should_end = True
        next_step = END

    else:
        # -------- 执行本轮所有工具调用 --------
        try:
            # 将 think_tool 与 ConductResearch 调用进行分类
            think_tool_calls = [
                tool_call for tool_call in most_recent_message.tool_calls
                if tool_call["name"] == "think_tool"
            ]

            conduct_research_calls = [
                tool_call for tool_call in most_recent_message.tool_calls
                if tool_call["name"] == "ConductResearch"
            ]

            # ---------- 执行 think_tool（同步执行） ----------
            for tool_call in think_tool_calls:
                observation = think_tool.invoke(tool_call["args"])
                tool_messages.append(
                    ToolMessage(
                        content=observation,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    )
                )

            # ---------- 执行 ConductResearch（并行异步） ----------
            if conduct_research_calls:
                # 并发启动多个 Research 子智能体
                coros = [
                    researcher_agent.ainvoke({
                        "researcher_messages": [
                            HumanMessage(content=tool_call["args"]["research_topic"])
                        ],
                        "research_topic": tool_call["args"]["research_topic"]
                    })
                    for tool_call in conduct_research_calls
                ]

                # 等待所有子研究任务完成
                tool_results = await asyncio.gather(*coros)

                # 将子智能体的研究结果封装为 ToolMessage
                # 每个子智能体返回 result["compressed_research"] 作为压缩后的研究结论
                # Supervisor 可通过 get_notes_from_tool_calls() 统一提取这些结果
                research_tool_messages = [
                    ToolMessage(
                        content=result.get(
                            "compressed_research",
                            "研究结果合成失败"
                        ),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    )
                    for result, tool_call in zip(tool_results, conduct_research_calls)
                ]

                tool_messages.extend(research_tool_messages)

                # 汇总所有子智能体返回的原始研究笔记
                all_raw_notes = [
                    "\n".join(result.get("raw_notes", []))
                    for result in tool_results
                ]

        except Exception as e:
            print(f"Supervisor 工具执行异常: {e}")
            should_end = True
            next_step = END

    # -------- 单一出口返回（Single Return Pattern） --------
    if should_end:
        return Command(
            goto=next_step,
            update={
                # 从所有 ToolMessage 中提取最终研究笔记
                "notes": get_notes_from_tool_calls(supervisor_messages),
                # 保留研究任务说明
                "research_brief": state.get("research_brief", "")
            }
        )
    else:
        return Command(
            goto=next_step,
            update={
                # 将工具执行结果写回 Supervisor 消息流
                "supervisor_messages": tool_messages,
                # 累积原始研究笔记
                "raw_notes": all_raw_notes
            }
        )


supervisor_builder = StateGraph(SupervisorState)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_edge(START, "supervisor")
supervisor_agent = supervisor_builder.compile()

async def main():
    research_brief = """我想要识别和评估中国东北三省（黑龙江、吉林、辽宁）被认为最好的咖啡店，特别是基于咖啡质量的评判。
    我的研究应集中于分析和比较这三省地区的咖啡店，以咖啡质量为主要标准。
    我对于评估咖啡质量的方法持开放态度（例如，专家评价、顾客评分、特色咖啡认证），
    除非直接影响咖啡质量的感知，否则对环境、位置、WiFi或食品选项没有限制。
    请优先使用主要来源，如咖啡店的官方网站、信誉良好的第三方咖啡评价机构
    （如Coffee Review或特色咖啡协会）以及像美团或大众点评等著名的评价聚合平台，
    在这些平台上可以找到关于咖啡质量的直接顾客反馈。
    该研究应得出一份有充分依据的中国东北三省顶级咖啡店列表或排名。"""
    result = await supervisor_agent.ainvoke({"supervisor_messages": [HumanMessage(content=f"{research_brief}.")]})
    format_messages(result['supervisor_messages'])

asyncio.run(main())