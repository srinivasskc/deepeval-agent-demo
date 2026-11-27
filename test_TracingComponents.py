import os

from deepeval.dataset import EvaluationDataset,Golden
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric
from deepeval.tracing import observe, update_current_trace
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import ToolCall
from deepeval.contextvars import get_current_golden
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from agent_instrumented import support_agent as _support_agent

load_dotenv()

class GroqJudge(DeepEvalBaseLLM):
    def load_model(self):
        return ChatGroq(
            model="openai/gpt-oss-120b",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )

    def generate(self, prompt, schema=None):
        model = (
            self.model.with_structured_output(schema, method="json_schema")
            if schema
            else self.model
        )
        response = model.invoke(prompt)
        return response if schema else response.content

    async def a_generate(self, prompt, schema=None):
        model = (
            self.model.with_structured_output(schema, method="json_schema")
            if schema
            else self.model
        )
        response = await model.ainvoke(prompt)
        return response if schema else response.content

    def get_model_name(self):
        return "openai/gpt-oss-120b"



# Trace we need actual values and expected values 
# We need to attach callback for agent and actual goldens.
@observe(name="support_agent")
def support_agent(user_input: str) -> str:
    golden = get_current_golden()
    if golden.expected_tools:
        update_current_trace(expected_tools=golden.expected_tools)
    if golden.expected_output:
        update_current_trace(expected_output=golden.expected_output)
    return _support_agent(user_input)


judge = GroqJudge()
taskCompletionMetric = TaskCompletionMetric(threshold=0.7, model=judge)
toolCorrectnessMetric = ToolCorrectnessMetric(model=judge)

dataset = EvaluationDataset(goldens=[
    Golden(input="Where is my order ORD-1042?", 
           expected_tools=[ToolCall(name="get_order_status")]),
    Golden(input="What is the return policy for electronics?",
           expected_tools=[ToolCall(name="get_refund_policy")])
])


for golden in dataset.evals_iterator(
    metrics=[taskCompletionMetric, toolCorrectnessMetric],
    async_config=AsyncConfig(run_async=False),
):
    support_agent(golden.input)

