# Fortune-500 Finance Multi-Agent System

A LangGraph-based multi-agent system that researches Fortune 500 companies,
retrieves their historical stock performance, and generates on-demand
visualizations — coordinated by a supervisor agent that turns three raw
tools into a single, coherent assistant.

This repository re-packages a hands-on LangGraph course (originally a set
of nine fill-in-the-blank Jupyter notebooks) into a proper, importable
Python project: shared code lives once in `src/`, the notebooks stay as
solved, runnable teaching material, and `main.py` gives you a working CLI
on top of it.

## Architecture

The system is built up in five stages, each a strictly more capable graph
over the same three tools. `main.py` defaults to the last one.

| Stage | Module | Description |
|---|---|---|
| 1 | `finance_agents.graphs.linear_agent` | Single agent. Every turn is routed through the tool node unconditionally. |
| 2 | `finance_agents.graphs.conditional_agent` | Single agent. `tools_condition` lets the LLM skip the tool node when no tool call is needed. |
| 3 | `finance_agents.graphs.react_agent` | Same behavior as stage 2, built in one line with `create_react_agent`. |
| 4 | `finance_agents.graphs.swarm_agent` | Two agents (`researcher`, `analyst`) that hand off to each other directly (peer-to-peer). |
| 5 | `finance_agents.graphs.supervisor_agent` | Two worker agents coordinated by a `supervisor`, which delegates tasks and composes the final answer. **This is the production architecture.** |

```
                       ┌──────────────┐
        user ────────► │  supervisor  │ ────────► single, composed response
                       └──────┬───────┘
                 delegates    │    delegates
               ┌──────────────┴──────────────┐
               ▼                              ▼
        ┌─────────────┐               ┌─────────────┐
        │  researcher  │               │   analyst   │
        │  - wikipedia │               │ - python repl│
        │  - stock_data│               └─────────────┘
        └─────────────┘
```

### Tools (`src/finance_agents/tools.py`)

| Tool | Purpose | Backing data |
|---|---|---|
| `wikipedia_tool` | Factual company summaries | Wikipedia API (no key required) |
| `stock_data_tool` | Historical stock performance | Local CSVs in `data/` (AAPL, MSFT, AMZN, META, NFLX, TSLA) |
| `python_repl_tool` | Executes generated Python to build charts (`plt.show()`) | LangChain's experimental `PythonREPL` |

All three tools are defined once in `tools.py` and imported by every graph —
in the original course material each notebook redefined all three tools
from scratch, which this project removes.

## Project structure

```
finance-multi-agent-system/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example
├── main.py                        # CLI entrypoint
├── data/                          # historical stock CSVs
├── docs/images/                   # architecture diagrams referenced above
├── src/finance_agents/
│   ├── tools.py                   # the 3 shared tools
│   ├── utils.py                   # pretty-printing for graph.stream() output
│   └── graphs/
│       ├── linear_agent.py
│       ├── conditional_agent.py
│       ├── react_agent.py
│       ├── swarm_agent.py
│       └── supervisor_agent.py
├── notebooks/                     # the original course, solved and renumbered
│   ├── 01_wikipedia_tool.ipynb
│   ├── 02_stock_data_tool.ipynb
│   ├── 03_python_repl_tool.ipynb
│   ├── 04_graph_basics.ipynb
│   ├── 05_linear_tool_calling_graph.ipynb
│   ├── 06_conditional_routing.ipynb
│   ├── 07_react_agent.ipynb
│   ├── 08_swarm_multi_agent.ipynb
│   └── 09_supervisor_multi_agent.ipynb
└── tests/
    └── test_tools.py              # deterministic, no-API-key smoke tests
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .                 # installs the finance_agents package from src/

cp .env.example .env             # only needed if Ollama isn't on localhost:11434
```

### Ollama (LLM backend)

The agents run against a local [Ollama](https://ollama.com) server via
`langchain-ollama` — no API key required.

```bash
# install Ollama, then:
ollama pull llama3.1        # or another tool-calling-capable model
ollama serve                # if not already running
```

Use `--model` / `-m` to point at a different local model tag (it must
support tool/function calling — `llama3.1`, `llama3.2`, and `qwen2.5` all
do; older Llama 2/3.0 tags generally don't).

## Usage

Interactive chat, supervisor architecture (default):

```bash
python main.py
```

Run a single question and exit:

```bash
python main.py -q "Has Tesla's share price increased over the last month?"
```

Try a different architecture:

```bash
python main.py --architecture swarm
python main.py --architecture react
```

Or run any stage as a standalone script, e.g.:

```bash
python -m finance_agents.graphs.supervisor_agent
```

### Example session

```
You: Who is Apple's CEO, and how has AAPL performed over the last week?
```

The supervisor delegates the CEO/company lookup and stock-data pull to
`researcher`, then returns a single combined answer — it never runs code
or answers from its own knowledge directly.

```
You: Plot AAPL's closing price over the last 30 days.
```

The supervisor hands this to `analyst`, which writes and executes the
matplotlib code via `python_repl_tool` and returns the rendered chart.

## Data notes

- Stock data is static, sourced from local CSVs (`data/*.csv`) rather than
  a live feed, so results reflect the last date present in each file, not
  today's market. Swapping `stock_data_tool` for a live API (e.g. Yahoo
  Finance) is a drop-in replacement — it only needs to keep the same
  `(company_ticker, num_days) -> markdown table` interface.
- Supported tickers: `AAPL`, `MSFT`, `AMZN`, `META`, `NFLX`, `TSLA`.

## Testing

```bash
pytest tests/
```

`tests/test_tools.py` exercises `stock_data_tool` and `python_repl_tool`
directly — no OpenAI API key or network access required. The agent graphs
themselves call an LLM and are best exercised interactively via `main.py`
or the notebooks.

## Extending this project

- **New tool**: add it to `ALL_TOOLS` in `tools.py`, then include it in
  whichever agent(s) in `graphs/` should have access to it.
- **New worker agent**: follow the pattern in `supervisor_agent.py` —
  build it with `create_react_agent`, give it a narrow tool set and a
  prompt that tells it to respond only to the supervisor, then add it to
  the `agents=[...]` list passed to `create_supervisor`.
- **Persistence beyond a single process**: swap `InMemorySaver` for one of
  LangGraph's persistent checkpointers (e.g. a Postgres or SQLite saver)
  to keep conversation state across restarts.

## Acknowledgements

Built on top of course material and diagrams originally distributed as a
DataCamp LangGraph course; this repository restructures that material into
a standalone, runnable engineering project.
