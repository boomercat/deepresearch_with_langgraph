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