from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
import os
from dotenv import load_dotenv
load_dotenv()
from langsmith import Client

# 场景 1：Agent 应该【继续】——信息不足或不相关 + think_tool 判断需要进一步研究
messages_should_continue = [
    HumanMessage(content="基于咖啡品质，旧金山有哪些顶级咖啡店？"),
    AIMessage(
        content="我将搜索有关旧金山咖啡店的信息，重点关注咖啡品质。",
        tool_calls=[
            {
                "name": "tavily_search",
                "args": {
                    "query": "旧金山 咖啡店 热门 地点"
                },
                "id": "call_continue_search_001",
                "type": "tool_call"
            }
        ]
    ),
    # Tavily 搜索结果仅提供了泛泛的咖啡馆信息——并未聚焦咖啡品质
    ToolMessage(
        content="搜索结果显示了旧金山一些受欢迎的咖啡馆：Blue Bottle Coffee 以良好的 WiFi 和氛围著称，Philz Coffee 因定制拼配和友好的服务而闻名，Café Reveille 在多个地点提供优质糕点和适合使用笔记本电脑的空间。Ritual Coffee 提供舒适的座位，深受学生欢迎。",
        tool_call_id="call_continue_search_001",
    ),
    # Agent 使用 think_tool 进行反思，意识到当前搜索结果不足
    AIMessage(
        content="我将使用 think_tool 来分析这些搜索结果，并确定下一步行动。",
        tool_calls=[
            {
                "name": "think_tool",
                "args": {
                    "reflection": "对搜索结果的分析：目前获取的信息主要集中在咖啡馆的一般特征，如 WiFi、氛围、定制拼配和座位条件，但缺乏具体的咖啡品质指标。用户明确要求基于咖啡品质进行评估，这通常需要专家评测、专业评分、精品咖啡认证，或专门关注咖啡品质的顾客反馈。我需要进一步搜索与咖啡品质相关的评估信息，例如 Coffee Review 评分、精品咖啡资质或针对旧金山咖啡店的专家评价。目前的信息不足以回答用户关于咖啡品质的问题。"
                },
                "id": "call_continue_think_001",
                "type": "tool_call"
            }
        ]
    ),
    # think_tool 的响应指出了信息缺口，并确认需要更有针对性的研究
    ToolMessage(
        content="反思已记录：对搜索结果的分析：目前获取的信息主要集中在咖啡馆的一般特征，如 WiFi、氛围、定制拼配和座位条件，但缺乏具体的咖啡品质指标。用户明确要求基于咖啡品质进行评估，这通常需要专家评测、专业评分、精品咖啡认证，或专门关注咖啡品质的顾客反馈。我需要进一步搜索与咖啡品质相关的评估信息，例如 Coffee Review 评分、精品咖啡资质或针对旧金山咖啡店的专家评价。目前的信息不足以回答用户关于咖啡品质的问题。",
        tool_call_id="call_continue_think_001",
    )
]

# 场景 2：Agent 应该【停止】——已获得全面的咖啡品质信息 + think_tool 反思确认
messages_should_stop = [
    HumanMessage(content="基于咖啡品质，旧金山有哪些顶级咖啡店？"),
    AIMessage(
        content="我将专门围绕咖啡品质评估和专家评价，研究旧金山的咖啡店。",
        tool_calls=[
            {
                "name": "tavily_search",
                "args": {
                    "query": "旧金山 咖啡 品质 最佳 精品咖啡 专家评测 Coffee Review 评分 2024"
                },
                "id": "call_stop_search_001", 
                "type": "tool_call"
            }
        ]
    ),
    # 该 ToolMessage 提供了【全面的】咖啡品质信息——足以回答问题
    ToolMessage(
        content="""针对旧金山的咖啡品质研究显示，以下咖啡店基于专家评测和精品咖啡资质表现突出：

**以咖啡品质为核心的顶级咖啡店：**
1. **Sightglass Coffee** —— Coffee Review 评分 94/100，以单一产地咖啡豆和精准烘焙技术著称，并因卓越的咖啡品质入选《Bon Appétit》“全美最佳咖啡”榜单。

2. **Saint Frank Coffee** —— 根据 The Daily Meal 基于咖啡品质的评选，被评为全美第 3 名咖啡店。Yelp 评分 4.3/5，评论中频繁提及“卓越的咖啡豆品质”和“完美的萃取”。

3. **Ritual Coffee Roasters** —— 精品咖啡协会（SCA）成员，采用直接贸易模式，Google 评分稳定在 4.4/5，顾客评价强调“出色的咖啡风味”和“专业的咖啡师技艺”。

4. **Blue Bottle Coffee** —— 第三波咖啡的先驱品牌，Coffee Review 平均评分在 90 分以上，以严格的品质控制和新鲜烘焙流程闻名。

5. **Four Barrel Coffee** —— 本地烘焙商，Coffee Review 评分区间为 88–92/100，因“卓越的咖啡品质”和创新的冲煮方式受到行业专家高度评价。

**识别出的咖啡品质指标包括：**
- Coffee Review 的专业评分（88–94/100）
- 精品咖啡协会认证
- 直接贸易 / 单一产地采购
- 来自美食媒体的专家认可
- 顾客评论中对咖啡风味与品质的直接评价

这些信息提供了一个**完全基于咖啡品质**（而非氛围或服务）的旧金山咖啡店综合评估。""",
        tool_call_id="call_stop_search_001",
    ),
    # Agent 使用 think_tool 对搜索结果进行反思
    AIMessage(
        content="我将使用 think_tool 对搜索结果进行反思，以评估是否已经获得足够的信息。",
        tool_calls=[
            {
                "name": "think_tool",
                "args": {
                    "reflection": "对搜索结果的分析：我已经获得了关于旧金山咖啡店的全面咖啡品质信息，包括：1）Coffee Review 的专业评分（88–94/100）；2）来自权威媒体的专家排名；3）精品咖啡协会认证；4）具体的品质指标（如单一产地、烘焙技术、萃取方法）；5）五家顶级咖啡店的详细品质评估。这些内容完整覆盖了用户提出的所有评估维度，包括专家评测、顾客评价和精品咖啡认证。我已经具备足够的信息，可以基于咖啡品质给出一个有充分依据的排名，不需要再进行额外搜索，现在可以直接给出完整答案。"
                },
                "id": "call_think_001",
                "type": "tool_call"
            }
        ]
    ),
    # think_tool 的响应，确认反思已被记录
    ToolMessage(
        content="反思已记录：对搜索结果的分析：我已经获得了关于旧金山咖啡店的全面咖啡品质信息，包括：1）Coffee Review 的专业评分（88–94/100）；2）来自权威媒体的专家排名；3）精品咖啡协会认证；4）具体的品质指标（如单一产地、烘焙技术、萃取方法）；5）五家顶级咖啡店的详细品质评估。这些内容完整覆盖了用户提出的所有评估维度，包括专家评测、顾客评价和精品咖啡认证。我已经具备足够的信息，可以基于咖啡品质给出一个有充分依据的排名，不需要再进行额外搜索，现在可以直接给出完整答案。",
        tool_call_id="call_think_001",
    )
]



langsmith_client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
dataset_name = "dresearch_subagent_test"
if not langsmith_client.has_dataset(dataset_name=dataset_name):

    # Create the dataset
    dataset = langsmith_client.create_dataset(
        dataset_name=dataset_name,
        description="A dataset that evaluates whether a researcher can accurately decide to continue calling tools, or to stop.",
    )
    
    # Create the examples
    langsmith_client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {"researcher_messages": messages_should_continue},
                "outputs": {"next_step": "continue"},
            },
            {
                "inputs": {"researcher_messages": messages_should_stop},
                "outputs": {"next_step": "stop"},
            },
        ],
    )

def evaluate_next_step(outputs: dict, reference_outputs: dict):
    tool_calls = outputs["researcher_messages"][-1].tool_calls
    made_tool_call = len(tool_calls) > 0
    return {
        "key": "correct_next_step",
        "score": made_tool_call == (reference_outputs["next_step"] == "continue")
    }

from langgraph_deepresearch.research_subagent import researcher_agent
import uuid
def target_func(inputs: dict):
    config = {"configurable": {"thread_id": uuid.uuid4()}}
    result = researcher_agent.nodes["llm_call"].invoke(inputs, config=config)
    return result

if __name__ == "__main__":
    langsmith_client.evaluate(
        target_func,
        data=dataset_name,
        evaluators=[evaluate_next_step],
        experiment_prefix="Researcher Iteration",
    )
