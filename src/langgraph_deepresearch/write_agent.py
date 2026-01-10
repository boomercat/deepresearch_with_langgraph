from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
import os
from dotenv import load_dotenv
load_dotenv()
from langgraph_deepresearch.utils import get_today_str
from langgraph_deepresearch.prompts import final_report_generation_prompt
from langgraph_deepresearch.state import AgentState, AgentInputState
from langgraph_deepresearch.clarify_subagent import (
    clarify_with_user,
    write_research_brief,
    human_review_research_brief,
)
from langgraph_deepresearch.supervisor_agent import supervisor_agent

from langchain.chat_models import init_chat_model
writer_model = init_chat_model(
    model=os.getenv("MODEL_NAME"),          # qwen-max / qwen2.5 / deepseek-r1
    model_provider="openai",                
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api-inference.modelscope.cn/v1/",
    max_tokens=32000,
    temperature=0.0,
)

async def final_report_generation(state: AgentState):
    """
    最终研究报告生成节点。

    负责将 Supervisor 阶段累计得到的所有研究笔记进行整合，
    并结合研究任务说明（research_brief），生成一份完整、
    结构化的最终研究报告。
    """

    # 从状态中获取已整理好的研究笔记
    notes = state.get("notes", [])

    # 将所有研究结论合并为单一文本，作为写作模型的输入素材
    findings = "\n".join(notes)

    # 构造最终报告生成 Prompt，注入研究背景、研究结论和当前日期
    final_report_prompt = final_report_generation_prompt.format(
        research_brief=state.get("research_brief", ""),
        findings=findings,
        date=get_today_str()
    )

    # 调用写作模型生成最终研究报告
    final_report = await writer_model.ainvoke(
        [HumanMessage(content=final_report_prompt)]
    )

    return {
        # 最终研究报告正文
        "final_report": final_report.content,

        # 用于消息流或 UI 展示的输出消息
        "messages": ["Here is the final report: " + final_report.content],
    }


deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

# Add workflow nodes
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("human_review_research_brief", human_review_research_brief)
deep_researcher_builder.add_node("supervisor_subgraph", supervisor_agent)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)

# Add workflow edges
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", "human_review_research_brief")
deep_researcher_builder.add_conditional_edges(
    "human_review_research_brief",
    lambda state: (
        "approved"
        if state.get("research_brief_approved")
        else "rejected"
        if state.get("research_brief_approved") is False
        else "pending"
    ),
    {
        "approved": "supervisor_subgraph",
        "rejected": "write_research_brief",
        "pending": END,
    },
)
deep_researcher_builder.add_edge("supervisor_subgraph", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)

# Compile the full workflow
agent = deep_researcher_builder.compile()
