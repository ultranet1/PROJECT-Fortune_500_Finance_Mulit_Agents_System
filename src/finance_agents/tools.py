"""
Tools available to the agents in this project.

Every agent in this project draws from the same toolbox:

1. ``wikipedia_tool``   -> factual company research (Wikipedia API)
2. ``stock_data_tool``  -> historical stock performance (local CSVs)
3. ``python_repl_tool`` -> arbitrary Python execution, used for plotting

Keeping the tools in one module means every graph in ``finance_agents.graphs``
imports the exact same implementation instead of re-defining it per notebook,
which is what the original course exercises did.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import pandas as pd
import wikipedia
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

# Resolve the data directory relative to this file so the tools work
# regardless of the caller's current working directory.
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

wikipedia.set_user_agent(
    "company-research-tool/1.0 (https://www.datacamp.com; content@datacamp.com)"
)


@tool
def wikipedia_tool(
    query: Annotated[str, "The Wikipedia search to execute to find key summary information."],
) -> str:
    """Use this to search Wikipedia for factual information."""
    try:
        # Step 1: Search using query
        results = wikipedia.search(query)

        if not results:
            return "No results found on Wikipedia."

        # Step 2: Retrieve page title
        title = results[0]

        # Step 3: Fetch summary
        summary = wikipedia.summary(title, sentences=8, auto_suggest=False, redirect=True)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\nWikipedia summary: {summary}"


@tool
def stock_data_tool(
    company_ticker: Annotated[str, "The ticker symbol of the company to retrieve their stock performance data."],
    num_days: Annotated[int, "The number of business days of stock data required to respond to the user query."],
) -> str:
    """
    Use this to look-up stock performance data for companies to retrieve a table from a CSV. You may need to
    convert company names into ticker symbols to call this function, e.g, Apple Inc. -> AAPL, and you may need
    to convert weeks, months, and years, into days.
    """

    # Load the CSV for the company requested
    file_path = DATA_DIR / f"{company_ticker}.csv"

    if not os.path.exists(file_path):
        return (
            f"Sorry, but data for company {company_ticker} is not available. "
            "Please try Apple, Amazon, Meta, Microsoft, Netflix, or Tesla."
        )

    stock_df = pd.read_csv(file_path, index_col="Date", parse_dates=True)

    # Ensure the index is in date format
    stock_df.index = stock_df.index.date

    # Maximum num_days supported by the dataset
    max_num_days = (stock_df.index.max() - stock_df.index.min()).days

    if num_days > max_num_days:
        return "Sorry, but this time period exceeds the data available. Please reduce it to continue."

    # Get the most recent date in the DataFrame
    final_date = stock_df.index.max()

    # Filter the DataFrame to get the last num_days of stock data
    filtered_df = stock_df[stock_df.index > (final_date - pd.Timedelta(days=num_days))]

    return (
        f"Successfully executed the stock performance data retrieval tool to retrieve the last "
        f"*{num_days} days* of data for company **{company_ticker}**:\n\n{filtered_df.to_markdown()}"
    )


_repl = PythonREPL()


@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
) -> str:
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user. The chart should be displayed
    using `plt.show()`."""
    try:
        result = _repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return (
        f"Successfully executed the Python REPL tool.\n\n"
        f"Python code executed:\n```python\n{code}\n```\n\n"
        f"Code output:\n```\n{result}```"
    )


ALL_TOOLS = [wikipedia_tool, stock_data_tool, python_repl_tool]
