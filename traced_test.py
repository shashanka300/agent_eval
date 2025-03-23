import os
from dotenv import load_dotenv
load_dotenv()

os.environ["TAVILY_API_KEY"] = os.getenv("API_KEY")

from langchain_ollama import ChatOllama
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch
from tracer import EnhancedUnifiedTracer

# Initialize tracer
tracer = EnhancedUnifiedTracer()

# Load model
local_llm = "qwen2.5:14b"
model = ChatOllama(model=local_llm, temperature=0.0)

# ========== TOOLS ==========
def add(a: float, b: float) -> float:
    '''Add two numbers.'''
    return a + b
add = tracer.trace(add, name_override="add", role="tool")


def multiply(a: float, b: float) -> float:
    '''Multiply two numbers.'''
    return a * b
multiply = tracer.trace(multiply, name_override="multiply", role="tool")


tavily_search_tool = tracer.trace(TavilySearch(), name_override="tavily_search", role="tool")

# ========== AGENTS ==========

math_agent = create_react_agent(
    model=model,
    tools=[add, multiply],
    name="math_expert",
    prompt="You are a math expert. Always use one tool at a time."
)
math_agent.invoke = tracer.trace(math_agent.invoke, name_override="math_agent",role="agent")

research_agent = create_react_agent(
    model=model,
    tools=[tavily_search_tool],
    name="research_expert",
    prompt="You are a world class researcher with access to web search. Do not do any math."
)
research_agent.invoke = tracer.trace(research_agent.invoke, name_override="research_agent", role="agent")
# ========== SUPERVISOR ==========

workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "Use research_agent for web queries and math_agent for calculations."
    )
)
app = workflow.compile()
app.invoke = tracer.trace(app.invoke, name_override="supervisor",role="supervisor agent")


# ========== RUN ==========
if __name__ == "__main__":
    querry = "What is 5 + 3?"
    tracer.set_current_query(querry)

    result = app.invoke({
        "messages": [
            {
                "role": "user",
                "content": querry
            }
        ]
    })

    for m in result["messages"]:
        m.pretty_print()

    print("\n--- TRACE SUMMARY ---")
    tracer.print_summary()
