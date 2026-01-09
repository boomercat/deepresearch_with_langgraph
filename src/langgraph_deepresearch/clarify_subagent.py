import os
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
from typing_extensions import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from langgraph_deepresearch.prompts import clarify_with_user_instructions, transform_messages_into_research_topic_prompt
from langgraph_deepresearch.state import AgentState, ClarifyWithUser, ResearchQuestion, AgentInputState
from langgraph_deepresearch.utils import format_messages

def get_today_str() -> str:
    return datetime.now().strftime("%a %b %d %Y")


model = init_chat_model(
    model=os.getenv("MODEL_NAME"),          # qwen-max / qwen2.5 / deepseek-r1
    model_provider="openai",                # 关键：强制走 OpenAI-compatible
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api-inference.modelscope.cn/v1/",
    temperature=0.0,
)


def clarify_with_user(state: AgentState) -> Command[Literal["write_research_brief", "__end__"]]:
    """
    判断用户请求的信息是否充足，决定是否可以继续生成研究简报。

    返回一个 Command，用于控制 LangGraph 中的执行流程：
    - 如果信息不足：向用户提出澄清问题，并结束当前流程
    - 如果信息充足：进入 write_research_brief 节点继续执行
    """   

    structured_output_model = model.with_structured_output(ClarifyWithUser, method="function_calling")

    response = structured_output_model.invoke([
        HumanMessage(content=clarify_with_user_instructions.format(
            messages=get_buffer_string(state["messages"]),
            date=get_today_str()
        ))])

    # 如果需要更多信息，本轮 agent 执行结束，等待用户下一轮输入
    if response.need_clarification:
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=response.question)]}
        )
    #反之，继续执行后续节点 write_research_brief
    else:
        return Command(
            goto="write_research_brief",
            update={"messages": [AIMessage(content=response.verification)]}
        )


def write_research_brief(state: AgentState):
    """
    将 message 转化为后续的研究摘要。
    使用结构化输出确保摘要符合所需格式，并包含有效研究所需的所有详细信息。
    """

    structured_output_model = model.with_structured_output(ResearchQuestion, method="function_calling")

    response = structured_output_model.invoke([
        HumanMessage(content=transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str()
        ))
    ])

    return {
        "research_brief": response.research_brief,
        "supervisor_messages": [AIMessage(content=response.research_brief)]
    }

# 连接图

scope_agent_builder = StateGraph(AgentState, input_schema=AgentInputState)

# Add workflow nodes
scope_agent_builder.add_node("clarify_with_user", clarify_with_user)
scope_agent_builder.add_node("write_research_brief", write_research_brief)

# Add workflow edges
scope_agent_builder.add_edge(START, "clarify_with_user")
scope_agent_builder.add_edge("write_research_brief", END)

# Compile the workflow
scope_research = scope_agent_builder.compile()

if __name__ == "__main__":
    thread = {"configuration": {"thread_id": "1"}}
    result = scope_research.invoke({
        "messages": [HumanMessage(content="东三省最好的学校是哪所？.")]
    })
    # format_messages(result)
    print(result["messages"])
    print(f"\n\n {result['research_brief']}")
