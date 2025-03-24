import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from traced_test import tracer
from run_agent_eval import run_agent_evaluation, test_cases, app

st.set_page_config(layout="wide")
sns.set_style("whitegrid")

# Run evaluations once
if "summary" not in st.session_state:
    run_agent_evaluation(app=app, tracer=tracer, test_cases=test_cases)
    st.session_state.summary = tracer.get_execution_summary()

summary = st.session_state.summary
call_data = summary["execution_summary"]

# Sidebar
st.sidebar.title("📊 Agent Evaluation Dashboard")
st.sidebar.markdown("Visualize agent routing, tool usage, and performance.")

# Title
st.title("🤖 Evaluation Summary Report")

# Tabs
tab1, tab2, tab3 = st.tabs(["📈 Overview", "🔗 Call Graphs", "📊 Evaluation Stats"])

with tab1:
    col1, col2 = st.columns(2)

    # Function Call Counts
    with col1:
        st.subheader("🔁 Function Call Frequency")
        df_counts = pd.DataFrame(list(call_data["function_call_counts"].items()), columns=["Function", "Count"])
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        sns.barplot(data=df_counts, x="Count", y="Function", palette="Blues_d", ax=ax1)
        ax1.set_xlabel("Calls")
        ax1.set_ylabel("")
        st.pyplot(fig1)

    # Function Execution Times
    with col2:
        st.subheader("⏱️ Avg. Execution Time (s)")
        df_times = pd.DataFrame([{"Function": k, **v} for k, v in call_data["function_execution_times"].items()])
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        sns.barplot(data=df_times, x="average", y="Function", palette="Greens_d", ax=ax2)
        ax2.set_xlabel("Avg Time (s)")
        ax2.set_ylabel("")
        st.pyplot(fig2)

    # Evaluation Table
    st.subheader("📊 Evaluation Sheet")
    from evaluators import ToolCallAccuracy, AgentRoutingAccuracy, EnhancedAgentGoalAccuracy
    tool_eval = ToolCallAccuracy()
    agent_eval = AgentRoutingAccuracy()
    goal_eval = EnhancedAgentGoalAccuracy()

    rows = []
    for case in test_cases:
        query = case["query"]
        expected_tool = case.get("expected_tool")
        expected_agent = case.get("expected_agent")
        expected_answer = case.get("expected_answer")

        actual_calls = tracer.query_function_calls.get(query, [])
        detected_tool = next((v for d in reversed(actual_calls) if "tool" in d for v in d.values()), None)
        detected_agent = next((v for d in actual_calls if "agent" in d for v in d.values()), None)

        tool_correct = tool_eval.evaluate_tool_call(query, detected_tool, {query.lower(): expected_tool or ""})
        agent_correct = agent_eval.evaluate_agent_routing(query, detected_agent, expected_agent)
        goal_result = goal_eval.evaluate_agent_goal_accuracy(query, expected_answer or "", tracer.execution_trace[-1] if tracer.execution_trace else "")

        rows.append({
            "Query": query,
            "Expected Agent": expected_agent,
            "Detected Agent": detected_agent,
            "Agent Correct": agent_correct,
            "Expected Tool": expected_tool,
            "Detected Tool": detected_tool,
            "Tool Correct": tool_correct,
            "Goal Achieved": goal_result
        })

    df_eval = pd.DataFrame(rows)
    st.dataframe(df_eval, use_container_width=True)

with tab2:
    st.subheader("🔗 Connected Graphs of Function Calls")
    call_order = call_data["function_call_order"]
    for i, flow in enumerate(call_order):
        net = Network(height='300px', width='100%', directed=True)
        st.markdown(f"**Query {i+1} Call Flow**")
        edges = list(zip(flow[:-1], flow[1:]))
        for node in set(flow):
            net.add_node(node, label=node)
        for src, dst in edges:
            net.add_edge(src, dst)
        net.save_graph(f"graph_{i}.html")
        HtmlFile = open(f"graph_{i}.html", 'r', encoding='utf-8')
        components.html(HtmlFile.read(), height=300)

with tab3:
    st.subheader("📊 Evaluation Summary Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✅ Agent Routing Accuracy", f"{agent_eval.calculate_accuracy() * 100:.2f}%")
    with col2:
        st.metric("⚒️ Tool Call Accuracy", f"{tool_eval.calculate_accuracy() * 100:.2f}%")
    with col3:
        st.metric("🎯 Goal Completion Accuracy", f"{goal_eval.calculate_accuracy() * 100:.2f}%")

# Raw summary
with st.expander("🛠 Full Execution Summary (JSON)"):
    st.json(summary)