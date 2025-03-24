from evaluators import (
    TopicAdherenceCalculator,
    ToolCallAccuracy,
    EnhancedAgentGoalAccuracy,
    AgentRoutingAccuracy
)
from traced_test import tracer
from traced_test import app

def run_agent_evaluation(app, tracer, test_cases: list):
    """
    Run a series of evaluation test cases on a compiled agent app.

    Each test case should be a dict:
    {
        "query": "What is 5 + 3?",
        "expected_tool": "add",
        "expected_agent": "math_agent",
        "expected_answer": "8",
        "topics": ["math", "science"]
    }
    """

    # Initialize evaluators
    topic_eval = TopicAdherenceCalculator()
    tool_eval = ToolCallAccuracy()
    goal_eval = EnhancedAgentGoalAccuracy()
    routing_eval = AgentRoutingAccuracy()

    for case in test_cases:
        query = case["query"]
        expected_tool = case.get("expected_tool")
        expected_agent = case.get("expected_agent")
        expected_answer = case.get("expected_answer")
        reference_topics = case.get("topics", [])

        tracer.set_current_query(query)

        result = app.invoke({
            "messages": [{"role": "user", "content": query}]
        })

        final_response = result["messages"][-1].content
        actual_calls = tracer.query_function_calls.get(query, [])

        # Extract tool and agent from trace
        detected_tool = next((v for d in reversed(actual_calls) if "tool" in d for v in d.values()), None)
        detected_agent = next((v for d in actual_calls if "agent" in d for v in d.values()), None)

        print(f"\n=== Running Test Case ===")
        print(f"Query: {query}")
        print(f"Expected Agent: {expected_agent}")
        print(f"Expected Tool: {expected_tool}")
        print(f"Expected Answer: {expected_answer}")
        print(f"Agent Response: {final_response}")
        print(f"Detected Tool: {detected_tool}")
        print(f"Detected Agent: {detected_agent}\n")

        # Evaluate
        topic_eval.check_topic_adherence(query, reference_topics, {"tool_called": detected_tool})
        routing_eval.evaluate_agent_routing(query, detected_agent, expected_agent)
        tool_eval.evaluate_tool_call(query, detected_tool, {query.lower(): expected_tool or ""})
        goal_eval.evaluate_agent_goal_accuracy(query, expected_answer, final_response)

    # Print all summaries
    print("\n=== TRACE SUMMARY ===")
    tracer.print_summary()

    print("\n=== EVALUATION SUMMARY ===")
    topic_eval.print_topic_adherence_summary()
    routing_eval.print_summary()
    tool_eval.print_tool_call_accuracy_summary()
    goal_eval.print_goal_accuracy_summary()



test_cases = [
    {
        "query": "What is 20 times 3??",
        "expected_tool": "multiply",
        "expected_agent": "math_agent",
        "expected_answer": "60",
        "topics": ["math"]
    },
    {
        "query": "Who is the president of the United States?",
        "expected_tool": "tavily_search",
        "expected_agent": "research_agent",
        "expected_answer": "joe biden",
        "topics": ["biography", "history"]
    },
    {
        "query": "What is the theory of relativity?",
        "expected_tool": None,
        "expected_agent": "research_agent",
        "expected_answer": "relativity",
        "topics": ["physics"]
    },
    {
        "query": "Add 15 and 20",
        "expected_tool": "add",
        "expected_agent": "math_agent",
        "expected_answer": "35",
        "topics": ["math"]
    },
    {
        "query": "Tell me about Newton’s laws of motion.",
        "expected_tool": "tavily_search",
        "expected_agent": "research_agent",
        "expected_answer": "motion",
        "topics": ["physics"]
    },
]

if __name__ == "__main__":
    run_agent_evaluation(app, tracer, test_cases)