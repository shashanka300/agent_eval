import functools
import time
import json
from typing import Callable, Any, Dict


class EnhancedUnifiedTracer:
    def __init__(self):
        self.call_counts = {}
        self.execution_times = {}
        self.call_order = []
        self.query_function_calls = {}
        self.current_query = None
        self.execution_trace = []

    def set_current_query(self, query: str):
        self.current_query = query
        self.query_function_calls[query] = []

    def _log(self, event_type: str, details: Any) -> None:
        trace_entry = f"[{event_type}] {details}"
        # print(trace_entry)
        self.execution_trace.append(trace_entry)

    def trace(self, target=None, *, name_override: str = None, role: str = None):
        def decorate(func: Callable, traced_name: str):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if self.current_query:
                    if role:
                        self.query_function_calls[self.current_query].append({role: traced_name})
                    else:
                        self.query_function_calls[self.current_query].append(traced_name)

                self.call_counts[traced_name] = self.call_counts.get(traced_name, 0) + 1
                self.call_order.append(traced_name)

                try:
                    input_summary = {
                        "args": [repr(a) for a in args],
                        "kwargs": {k: repr(v) for k, v in kwargs.items()}
                    }
                except Exception:
                    input_summary = "Unserializable inputs"

                self._log("FUNCTION START", f"{traced_name} | Inputs: {input_summary}")

                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    elapsed = time.perf_counter() - start_time
                    self.execution_times.setdefault(traced_name, []).append(elapsed)
                    self._log("ERROR", f"{traced_name} failed after {elapsed:.4f}s: {e}")
                    raise

                elapsed = time.perf_counter() - start_time
                self.execution_times.setdefault(traced_name, []).append(elapsed)

                try:
                    output_summary = repr(result)
                except Exception:
                    output_summary = "Unserializable output"

                self._log("FUNCTION END", f"{traced_name} | Output: {output_summary} | Time: {elapsed:.4f}s")
                return result

            return wrapper

        # Case 1: Direct use: tracer.trace(fn)
        if callable(target) and not hasattr(target, "invoke"):
            traced_name = name_override or target.__name__
            return decorate(target, traced_name)

        # Case 2: Used as decorator: @tracer.trace(...)
        if target is None:
            def wrapper_decorator(fn):
                traced_name = name_override or fn.__name__
                return decorate(fn, traced_name)
            return wrapper_decorator

        # Case 3: Tool object with .invoke (e.g. TavilySearch)
        elif hasattr(target, "invoke"):
            traced_name = name_override or target.__class__.__name__
            docstring = getattr(target, "__doc__", f"Tool: {traced_name}")

            def wrapped_tool(input: str) -> str:
                return target.invoke(input)

            wrapped_tool.__name__ = traced_name
            wrapped_tool.__doc__ = docstring
            wrapped_tool.__annotations__ = {"input": str, "return": str}

            return decorate(wrapped_tool, traced_name)

        raise TypeError("Unsupported input to tracer.trace()")

    def get_query_function_calls(self) -> Dict[str, Any]:
        return self.query_function_calls

    def get_execution_summary(self) -> Dict:
        function_execution_times = {
            fname: {
                "total": round(sum(times), 4),
                "average": round(sum(times) / len(times), 4),
                "calls": len(times)
            }
            for fname, times in self.execution_times.items()
        }

        return {
            "execution_summary": {
                "function_call_counts": self.call_counts,
                "function_execution_times": function_execution_times,
                "function_call_order": self.call_order.copy(),
                "query_function_calls": self.get_query_function_calls()
            }
        }

    def get_final_trace(self) -> Dict:
        final_output = self.get_execution_summary()
        final_output["detailed_execution_trace"] = self.execution_trace
        return final_output

    def print_summary(self) -> None:
        summary = self.get_execution_summary()
        print("\n=== Execution Summary ===")
        print(json.dumps(summary, indent=2))
