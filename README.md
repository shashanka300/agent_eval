# Agent Evaluation Framework

This project evaluates agent-based workflows using an LLM-based judge (Ollama) and a tracer that logs detailed execution traces.

---

## ✅ Features

### 📊 Tracer
The `EnhancedUnifiedTracer` logs:
- Function call order
- Execution time per call
- Calls per query
- Agent/tool usage
- Supports per-query breakdown and structured summary

After evaluation, a full trace summary is printed:
```bash
=== Execution Summary ===
{
  "function_call_counts": { ... },
  "function_execution_times": { ... },
  "query_function_calls": { ... }
}
```

### 🧠 LLM Evaluators
Located in `llm_evaluators.py`, this module provides:
- `judge_tool_call(query, tool)` → Returns (True/False, reasoning)
- `judge_routing(query, agent)` → Returns (True/False, reasoning)
- `judge_from_tracer(tracer, query)` → Automatically extracts tool/agent from trace and evaluates both

Uses local LLM via Ollama (e.g., `qwen2.5:14b`) to determine correctness of routing and tool usage.

---

Built and evaluated using LangGraph + LangChain + Ollama by [You].

