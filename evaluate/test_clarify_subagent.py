from langchain_core.messages import HumanMessage, AIMessage
import os
from langsmith import Client
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langgraph_deepresearch.src.prompts import BRIEF_CRITERIA_PROMPT, BRIEF_HALLUCINATION_PROMPT
from langgraph_deepresearch.src.clarify_subagent import scope_agent_builder
# 测试用例
#---------------------------
# 测试用例1
conversation_1 = [
    HumanMessage(content="我想为退休投资5万美元，最好的投资方式是什么？"),
    AIMessage(content=(
        "为了更好地根据您的退休目标（5万美元）提供投资建议，"
        "我需要一些补充信息：\n"
        "1. 您目前的年龄或计划退休的年龄\n"
        "2. 您的风险承受能力（低 / 中 / 高）\n"
        "3. 是否对某些投资类型有偏好（例如：股票、债券、指数基金、房地产等）\n"
        "4. 您是通过税收优惠账户（如 IRA、401(k)）投资，还是通过普通的证券账户\n"
        "这些信息将有助于我为您提供更个性化、更有针对性的建议。"
    )),
    HumanMessage(content=(
        "我今年25岁，希望在45岁左右退休。目前的风险承受能力较高，"
        "但我认为随着年龄增长会逐渐降低。我听说股票和ETF是不错的选择，"
        "但我也愿意考虑其他类型的投资。另外，我已经有一个401(k)，"
        "不过这笔钱将通过普通的证券账户进行投资。"
    )),
]

# 测试用例2
conversation_2 = [
    HumanMessage(content="我正在纽约找公寓，可以帮我吗？"),
    AIMessage(content=(
        "为了更好地帮您筛选合适的纽约公寓，能否请您提供以下偏好信息：\n"
        "1. 希望居住的区域或街区\n"
        "2. 房间数量（卧室 / 卫生间）\n"
        "3. 可接受的月租预算范围\n"
        "4. 是否有必须具备的设施或功能\n"
        "5. 计划的入住时间\n"
        "这些信息将帮助我为您推荐更符合需求的公寓。"
    )),
    HumanMessage(content=(
        "我更倾向于住在切尔西（Chelsea）、熨斗区（Flatiron）或西村（West Village）。"
        "我希望是两室两卫，月租预算在7000美元以下。"
        "最好是有门卫的公寓楼，并且房间内配有洗衣机和烘干机，"
        "如果没有也可以接受。如果大楼有健身房会是加分项。"
        "我希望在2025年9月入住。"
    )),
]

# 测试用例1的标准回答
criteria_1 = [
    "当前年龄为25岁",
    "期望的退休年龄为45岁",
    "当前的风险承受能力较高",
    "有意投资于股票和ETF",
    "对股票和ETF以外的其他投资形式持开放态度",
    "投资账户为普通证券账户",
]

# 测试用例2的标准回答
criteria_2 = [
    "寻找位于切尔西（Chelsea）、熨斗区（Flatiron）或西村（West Village）的两室两卫公寓",
    "月租预算低于7000美元",
    "公寓应为有门卫的楼宇",
    "最好配备室内洗衣机和烘干机，但不是硬性要求",
    "最好配有健身房，但不是硬性要求",
    "计划入住时间为2025年9月",
]


# 数据集初始化
# -------------------------

langsmith_client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))

dataset_name = "clarify_subagent_test"
if not langsmith_client.has_dataset(dataset_name):
    langsmith_client.create_dataset(dataset_name)

    dataset = langsmith_client.create_dataset(
        dataset_name=dataset_name,
        description="测量研究摘要的质量的数据集")

    langsmith_client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {"messages": conversation_1},
                "outputs": {"messages": criteria_1}
            },
            {
                "inputs": {"messages": conversation_2},
                "outputs": {"messages": criteria_2}
            }
        ]
    )


#评估维度1 针对成功标准的覆盖度
#----------------------
class Criteria(BaseModel):
    """
    单条「成功标准」的评估结果模型。

    这个模型表示：
    - 某一条具体的成功标准（criteria）
    - 研究简报中是否覆盖了这条标准
    - 判断的详细理由（基于简报内容）
    """
    criteria_text: str = Field(
        description="具体的成功标准维度文本，之前我们自己给的（例如：'当前年龄是 25 岁'、'月租低于 7000'）"
    )

    reasoning: str = Field(
        description="判断该标准是否被覆盖的详细理由，需要引用研究简报中的具体内容作为依据"
    )

    is_captured: bool = Field(
        description="该成功标准是否在研究简报中被充分体现（True 表示满足，False 表示缺失或不足）"
    )


def evaluate_success_criteria(outputs: dict, reference_outputs: dict):
    """
    评估研究简报是否满足所有成功标准。

    该函数会：
    - 逐条检查每一个成功标准
    - 对每条标准给出是否满足的判断 + 理由
    - 最终计算一个总体得分（0.0 ~ 1.0）

    Args:
        outputs:
            包含模型生成结果的字典，必须包含：
            - outputs["research_brief"]：待评估的研究简报文本

        reference_outputs:
            包含评估参考标准的字典，必须包含：
            - reference_outputs["criteria"]：成功标准列表

    Returns:
        一个字典，包含：
        - key：评估指标名称
        - score：总体评分（满足的标准数 / 总标准数）
        - individual_evaluations：每条标准的详细评估结果
    """
    research_brief = outputs["research_brief"]
    success_criteria = reference_outputs["criteria"]

    model = init_chat_model(
        model=os.getenv("MODEL_NAME"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api-inference.modelscope.cn/v1/",
        temperature=0.0,
    )

    structured_output_model = model.with_structured_output(Criteria, method="function_calling")
    
    responses = structured_output_model.batch([
    [
        HumanMessage(
            content=BRIEF_CRITERIA_PROMPT.format(
                research_brief=research_brief,
                criterion=criterion
            )
        )
    ] 
    for criterion in success_criteria])

    individual_evaluations = [
        Criteria(
            reasoning=response.reasoning,
            criteria_text=criterion,
            is_captured=response.is_captured
        )
        for criterion, response in zip(success_criteria, responses)
    ]

    captured_count = sum(1 for eval_result in individual_evaluations if eval_result.is_captured)
    total_count = len(individual_evaluations)

    return {
        "key": "success_criteria_score", 
        "score": captured_count / total_count if total_count > 0 else 0.0,
        "individual_evaluations": [
            {
                "criteria": eval_result.criteria_text,
                "captured": eval_result.is_captured,
                "reasoning": eval_result.reasoning
            }
            for eval_result in individual_evaluations
        ]
    }




#评估维度2 针对研究简报是否做了不该做的假设
#-----------------------
class NoAssumptions(BaseModel):
    """
    用于评估「研究简报是否做了不该做的假设」的结构化输出模型。

    该模型检查 research brief 中是否包含：
    - 用户原对话里没有明确说过的信息
    - 额外的推断/臆测/补充要求（hallucination 风险）
    并输出详细理由，说明为什么判定为“无假设”或“有假设”。
    """


    no_assumptions: bool = Field(
        description=(
            "研究简报是否避免了不合理的假设。"
            "如果简报只包含用户明确提供的信息，则为 True；"
            "如果简报加入了用户未明确说明的内容/偏好/要求，则为 False。"
        )
    )

    reasoning: str = Field(
        description=(
            "对判定结果的详细解释。"
            "如果发现了假设，需要给出具体例子；"
            "如果没有假设，需要确认简报内容都来自用户明确陈述。"
        )
    )



def evaluate_no_assumptions(outputs: dict, reference_outputs: dict):
    """
    评估 research brief 是否避免做“用户没说过”的假设。

    这个 evaluator 的核心目标是：
    - 检查简报是否只包含用户明确提供的事实和要求
    - 不要凭空添加用户未提到的偏好/限制/背景信息

    Args:
        outputs:
            包含待评估文本的字典，必须包含：
            - outputs["research_brief"]：要检查的研究简报

        reference_outputs:
            参考信息字典，通常包含：
            - reference_outputs["criteria"]：成功标准列表（这里当作参照上下文传给 prompt）

    Returns:
        返回一个字典，包含：
        - key：评估项名称
        - score：布尔分数（True/False）
        - reasoning：解释理由
    """

    research_brief = outputs["research_brief"]
    success_criteria = reference_outputs["criteria"]

    model = init_chat_model(
        model=os.getenv("MODEL_NAME"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api-inference.modelscope.cn/v1/",
        temperature=0.0,
    )

    structured_output_model = model.with_structured_output(NoAssumptions, method="function_calling")

    response = structured_output_model.invoke([
        HumanMessage(
            content=BRIEF_HALLUCINATION_PROMPT.format(
                research_brief=research_brief,
                success_criteria=str(success_criteria)
            ))
    ])
    
    return {
        "key": "no_assumptions_score",
        "score": response.no_assumptions,
        "reasoning": response.reasoning
    }


#--------------------
# 开始评估

if __name__ == "__main__":
    import uuid
    from langgraph.checkpoint.memory import InMemorySaver

    scope = scope_agent_builder(InMemorySaver())

    def target_func(inputs: dict):
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        return scope.invoke(inputs, config=config)

    langsmith_client.evaluate(
        target_func,
        data=dataset_name,
        evaluators=[evaluate_success_criteria, evaluate_no_assumptions],
        experiment_prefix="test clarify subagent scope"
    )
