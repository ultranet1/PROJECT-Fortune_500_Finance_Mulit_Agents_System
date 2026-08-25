"""
finance_agents
==============

A small multi-agent system, built with LangGraph, that researches
Fortune-500 companies, pulls their historical stock performance, and
generates visualizations on request.

The package is organized so each stage of the underlying course
maps to an importable, testable module instead of a notebook cell:

- ``tools``                 -> the three tools shared by every agent
- ``utils``                 -> pretty-printing helpers for graph streams
- ``graphs.linear_agent``   -> single agent, unconditional tool routing
- ``graphs.conditional_agent`` -> single agent, conditional tool routing
- ``graphs.react_agent``    -> single agent, built with create_react_agent
- ``graphs.swarm_agent``    -> two agents, peer-to-peer handoffs (swarm)
- ``graphs.supervisor_agent`` -> two agents, coordinated by a supervisor
"""

from importlib import metadata

try:
    __version__ = metadata.version("finance-agents")
except metadata.PackageNotFoundError:  # running from source, not installed
    __version__ = "0.1.0"
