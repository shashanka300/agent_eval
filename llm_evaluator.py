from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from typing import Tuple
import re

# Set up the local LLM (Ollama)
llm = ChatOllama(model="qwen2.5:14b", temperature=0.0)

def extract_binary_flag(text: str) -> int:
    """Extract a 1 or 0 from the LLM response."""
    match = re.search(r"\b(1|0)\b", text.strip())
    return int(match.group(1)) if match else 0

def judge_tool_call(query: str, tool: str) -> Tuple[bool, str]:
    """
    Ask the LLM whether the tool used is appropriate for the query.
    Returns (is_correct, reasoning).
    """
    prompt = f"""
    Query: "{query}"
    Tool Used: {tool}

    Respond with 1 if this is the correct tool for the query, otherwise respond with 0.
    After the number, briefly explain why.
    Example:
    1 - Because it is a factual lookup and Wikipedia is appropriate.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    raw_content = response.content.strip()
    is_correct = extract_binary_flag(raw_content) == 1
    return is_correct, raw_content

def judge_routing(query: str, agent: str) -> Tuple[bool, str]:
    """
    Ask the LLM whether the agent chosen is appropriate for the query.
    Returns (is_correct, reasoning).
    """
    prompt = f"""
    Query: "{query}"
    Agent Routed To: {agent}

    Respond with 1 if this is the correct agent for the query, otherwise respond with 0.
    After the number, briefly explain why.
    Example:
    1 - This is a math problem, so math_expert is appropriate.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    raw_content = response.content.strip()
    is_correct = extract_binary_flag(raw_content) == 1
    return is_correct, raw_content

def judge_from_tracer(tracer, query: str) -> Tuple[Tuple[bool, str], Tuple[bool, str]]:
    """
    Evaluate tool and agent routing accuracy using the tracer logs and LLM as judge.
    """
    calls = tracer.query_function_calls.get(query, [])

    # Infer tool (typically the last call)
    tool_used = calls[-1] if calls else "Unknown"

    # Infer agent (first non-supervisor function after root call)
    agent_used = calls[1] if len(calls) > 1 else "Unknown"

    tool_ok, tool_reason = judge_tool_call(query, tool_used)
    routing_ok, route_reason = judge_routing(query, agent_used)

    return (tool_ok, tool_reason), (routing_ok, route_reason)

# Example usage
def test_llm_evaluators():
    from traced_test import app, tracer

    test_queries = [
        "What is the theory of relativity?",
        "What is 14 plus 5?",
        "Tell me about climate change."
    ]

    for query in test_queries:
        tracer.set_current_query(query)
        app.invoke({"messages": [{"role": "user", "content": query}]})

        calls = tracer.query_function_calls.get(query, [])
        tool_used = calls[-1] if calls else "Unknown"
        agent_used = calls[1] if len(calls) > 1 else "Unknown"

        tool_result, routing_result = judge_from_tracer(tracer, query)

        print(f"\nQuery: {query}")
        print(f"Agent Used: {agent_used}")
        print(f"Tool Used: {tool_used}")
        print("Tool Call Accuracy:", tool_result)
        print("Agent Routing Accuracy:", routing_result)


if __name__ == "__main__":
    test_llm_evaluators()