import operator
from typing_extensions import Optional, List, Annotated, TypedDict, Sequence

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState

class AgentInputState(MessagesState):
   """
   State for the agent input.
   """
   pass

class AgentState(MessagesState):
    """
    full multi-agent的主要state。
    在 MessagesState 的基础上扩展了用于研究的附加字段。
    为了在 subgraph 和主工作流之间正确管理状态，某些字段在不同的状态类中存在重复   
    """
   
    #研究摘要 从user的输入中生成
    research_brief: Optional[str]
    # 与supervisor的消息
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    #未整理的消息，从subagent中收集
    raw_notes: Annotated[list[str], operator.add] = []
    # 整理后的消息，用于生成最终报告
    notes: Annotated[list[str], operator.add] = []
    #最终报告
    final_report: str


# ===== STRUCTURED OUTPUT SCHEMAS =====

class ClarifyWithUser(BaseModel):
    """ 针对用户想要研究的相关内容是否需要反问
        如果需要则need_clarification=True
        同时添加question=xxx。
        如果不需要则need_clarification=False
        附加verification=xxx。
    """
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )
    
class ResearchQuestion(BaseModel):
    """ 生成的研究摘要格式"""
    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )


class ResearcherState(TypedDict):
    """
    research Agent 的 State。

    这个 State 用来保存整个研究过程中需要持续传递的数据，包括：
    - 对话历史（researcher_messages）
    - 已经调用工具的次数（用于限制循环）
    - 当前研究主题
    - 压缩后的研究结论
    - 详细的原始研究笔记
    """

    # 研究 Agent 的对话消息历史
    # 使用 add_messages 表示：每次节点返回的 messages 会自动追加到这里
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    # 工具调用的轮数 / 次数，用于防止无限循环
    tool_call_iterations: int
    # 当前正在研究的主题（由用户需求或澄清阶段确定）
    research_topic: str
    # 对研究结果的“压缩总结版本”，用于后续生成报告或回答
    compressed_research: str
    # 原始研究笔记列表（例如：网页摘要、工具返回结果等）
    # 使用 operator.add 表示多个节点产生的 notes 会合并到一个 list 中
    raw_notes: Annotated[List[str], operator.add]
    

class ResearcherOutputState(TypedDict):
    """
    researcher Agent 的最终输出state。

    表示整个研究流程结束后，对外返回的结果，通常包括：
    - 压缩后的研究结论
    - 所有原始研究笔记
    - 最终保留下来的对话记录
    """

    # 压缩后的研究结论（最终对用户有价值的总结）
    compressed_research: str

    # 研究过程中产生的全部原始笔记（未压缩）
    raw_notes: Annotated[List[str], operator.add]

    # 研究 Agent 的对话历史（最终状态）
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]

class Summary(BaseModel):
    """
    网页或文档内容的结构化摘要模型。

    通常用于：
    - 对搜索结果页面进行摘要
    - 提取关键内容，减少上下文长度
    """

    # 对网页或文档内容的简要总结
    summary: str = Field(
        description="网页内容的简明摘要。",
    )

    # 从内容中摘取的重要原文片段或关键引用
    key_excerpts: str = Field(
        description="内容中的关键引述或重要片段。",
    )



class SupervisorState(TypedDict):
    """
    multiagent  supervisor state。

    用于管理 Supervisor 与多个 Research 子智能体之间的协作流程，
    跟踪研究进度，并汇总来自各个子智能体的研究成果。
    """

    # Supervisor 与系统 / 子智能体之间用于决策和协调的消息记录
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]

    # 研究总纲 / 研究任务说明，用于指导整体研究方向
    research_brief: str

    # 已整理、结构化后的研究笔记，用于最终报告生成
    notes: Annotated[list[str], operator.add] = []

    # 研究迭代次数计数器，记录已执行的研究轮数
    research_iterations: int = 0

    # 来自各个子智能体的原始研究笔记（尚未整理）
    raw_notes: Annotated[list[str], operator.add] = []


@tool
class ConductResearch(BaseModel):
    """
    研究任务下发工具。

    用于由 Supervisor 将具体研究任务委派给某个专业 Research 子智能体。
    """

    research_topic: str = Field(
        description=(
            "需要研究的主题。必须是一个单一且明确的研究主题，"
            "并且需要以较高的详细程度进行描述（至少一个完整段落）。"
        ),
    )


@tool
class ResearchComplete(BaseModel):
    """
    研究完成信号工具。

    用于通知 Supervisor：当前研究流程已经完成，
    可以进入结果汇总或最终报告生成阶段。
    """
    pass
    