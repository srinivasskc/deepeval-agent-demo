import os

from deepeval.test_case import LLMTestCase
from deepeval.evaluate import evaluate
from deepeval.metrics import TaskCompletionMetric
from deepeval.models import DeepEvalBaseLLM
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from agent_instrumented import support_agent

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


actual_result = support_agent("Where is my order ORD-1042?")

# Create DataSet
test_case1 = LLMTestCase(
    input="Where is my order ORD-1042?", actual_output=actual_result
)

# BlackBox Evaluation

# Initialize metric - If metric is 0.8 and above - Test will pass
task_completion = TaskCompletionMetric(threshold=0.8, model=GroqJudge())

evaluate(
    test_cases=[test_case1],
    metrics=[task_completion]
    # identifier=os.getenv("DEEPEVAL_RUN_IDENTIFIER", "task-completion-groq"),
)
