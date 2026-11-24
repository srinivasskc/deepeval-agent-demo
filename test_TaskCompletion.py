from deepeval.test_case import LLMTestCase
from deepeval.evaluate import evaluate
from deepeval.metrics import TaskCompletionMetric
from agent_instrumented import support_agent

actual_result = support_agent("Where is my order ORD-1042?")

# Create DataSet
test_case1 = LLMTestCase(
    input="Where is my order ORD-1042?", 
    actual_output=actual_result
)

# Initialize metric - If metric is 0.8 and above - Test will pass
task_completion = TaskCompletionMetric(threshold=0.8, model="gpt-4o")

evaluate(test_cases = [test_case1], 
         metrics= [task_completion])
