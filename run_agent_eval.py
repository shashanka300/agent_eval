from evaluators import (
    TopicAdherenceCalculator,
    ToolCallAccuracy,
    EnhancedAgentGoalAccuracy
)
from traced_test import tracer
from traced_test import app

def run_agent_evaluation(app, tracer, test_cases: list):
    """
    Run a series of evaluation test cases on a compiled agent app.
    
    Each test case should be a dict:
    {
        "query": "What is 5 + 3?",
        "expected_tool": "math_expert",
        "expected_answer": "8",
        "topics": ["math", "science"]
    }
    """

    # Initialize evaluators
    topic_eval = TopicAdherenceCalculator()
    tool_eval = ToolCallAccuracy()
    goal_eval = EnhancedAgentGoalAccuracy()

    for case in test_cases:
        query = case["query"]
        expected_tool = case["expected_tool"]
        expected_answer = case["expected_answer"]
        reference_topics = case.get("topics", [])

        tracer.set_current_query(query)

        result = app.invoke({
            "messages": [{"role": "user", "content": query}]
        })

        # Get final response message
        final_response = result["messages"][-1].content

        # Detect last tool used (from tracer logs)
        query_key = query
        actual_calls = tracer.query_function_calls.get(query_key, [])

        last_tool = actual_calls[-1] if len(actual_calls) > 1 else None
        print(f"\n=== Running Test Case ===")
        print(f"Query: {query}")
        print(f"Expected Tool: {expected_tool}")
        print(f"Expected Answer: {expected_answer}")
        print(f"Agent Response: {final_response}")
        print(f"Detected Tool: {last_tool}\n")

        # Evaluate
        topic_eval.check_topic_adherence(query, reference_topics, {"tool_called": last_tool})
        tool_eval.evaluate_tool_call(query, last_tool, {
            query.lower(): expected_tool
        })
        goal_eval.evaluate_agent_goal_accuracy(query, expected_answer, final_response)

    # Print all summaries
    print("\n=== TRACE SUMMARY ===")
    tracer.print_summary()

    print("\n=== EVALUATION SUMMARY ===")
    topic_eval.print_topic_adherence_summary()
    tool_eval.print_tool_call_accuracy_summary()
    goal_eval.print_goal_accuracy_summary()

if __name__ == "__main__":
    
    test_cases = [
    {
        "query": "What is 19 times 3?",
        "expected_tool": "math_agent",
        "expected_answer": "57",
        "topics": ["math"]
    },
    {
        "query": "Who is the president of the United States?",
        "expected_tool": "research_agent",
        "expected_answer": "joe biden",  # case-insensitive match
        "topics": ["biography", "history"]
    },
    {
        "query": "What is the theory of relativity?",
        "expected_tool": "research_agent",
        "expected_answer": "relativity",
        "topics": ["physics"]
    },
    {
        "query": "Add 15 and 20",
        "expected_tool": "math_agent",
        "expected_answer": "35",
        "topics": ["math"]
    },
    {
        "query": "Tell me about Newton’s laws of motion.",
        "expected_tool": "research_agent",
        "expected_answer": "motion",
        "topics": ["physics"]
    },
]

run_agent_evaluation(app, tracer, test_cases)

