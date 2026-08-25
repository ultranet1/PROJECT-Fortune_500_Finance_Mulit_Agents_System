"""
Smoke tests for the deterministic, non-LLM parts of the project (the
tools themselves). These don't require an OpenAI API key and don't hit
the network for the stock tool, so they're safe to run in CI.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finance_agents.tools import python_repl_tool, stock_data_tool  # noqa: E402


def test_stock_data_tool_returns_markdown_table():
    result = stock_data_tool.invoke({"company_ticker": "AAPL", "num_days": 5})
    assert "Successfully executed" in result
    assert "|" in result  # markdown table


def test_stock_data_tool_unknown_ticker():
    result = stock_data_tool.invoke({"company_ticker": "NOPE", "num_days": 5})
    assert "not available" in result


def test_python_repl_tool_executes_code():
    result = python_repl_tool.invoke({"code": "print(1 + 1)"})
    assert "2" in result


def test_python_repl_tool_reports_errors():
    # PythonREPL.run() captures exceptions internally rather than raising,
    # so a bad snippet still comes back as a "successfully executed" tool
    # call whose *output* contains the traceback/error, not our own
    # `except BaseException` branch (which only fires if PythonREPL itself
    # misbehaves).
    result = python_repl_tool.invoke({"code": "1 / 0"})
    assert "ZeroDivisionError" in result
