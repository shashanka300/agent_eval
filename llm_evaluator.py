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
    Returns (is_correct, raw_response).
    """
    if tool == "Unknown":
        return False, "0"  # Tool not used

    prompt = f"""
    Query: "{query}"
    Tool Used: {tool}

    Respond only with 1 if this is the correct tool for the query, otherwise respond with 0.
    Do not explain.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    raw_content = response.content.strip()
    is_correct = extract_binary_flag(raw_content) == 1
    return is_correct, raw_content

def judge_routing(query: str, agent: str) -> Tuple[bool, str]:
    """
    Ask the LLM whether the agent routed is appropriate for the query.
    Returns (is_correct, raw_response).
    """
    if agent == "Unknown":
        return False, "0"

    prompt = f"""
    Query: "{query}"
    Agent Routed To: {agent}

    Respond only with 1 if this is the correct agent for the query, otherwise respond with 0.
    Do not explain.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    raw_content = response.content.strip()
    is_correct = extract_binary_flag(raw_content) == 1
    return is_correct, raw_content

def judge_from_tracer(tracer, query: str) -> Tuple[bool, bool]:
    """
    Evaluate tool and agent routing accuracy using the tracer logs and LLM as judge.
    Returns (tool_ok, routing_ok)
    """
    calls = tracer.query_function_calls.get(query, [])

    agent_used = "Unknown"
    tool_used = "Unknown"

    for entry in calls:
        if "agent" in entry:
            agent_used = entry["agent"]
            break

    for entry in reversed(calls):
        if "tool" in entry:
            tool_used = entry["tool"]
            break

    print(f"Detected Agent: {agent_used}, Tool: {tool_used}")

    tool_ok, _ = judge_tool_call(query, tool_used)
    routing_ok, _ = judge_routing(query, agent_used)

    return tool_ok, routing_ok

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

        tool_ok, routing_ok = judge_from_tracer(tracer, query)

        print(f"\nQuery: {query}")
        print("Tool Call Accuracy:", int(tool_ok))
        print("Agent Routing Accuracy:", int(routing_ok))


if __name__ == "__main__":
    test_llm_evaluators()
