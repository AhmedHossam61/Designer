import os
from typing import TypedDict, Sequence, Literal
from typing_extensions import Annotated
import json
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import PyPDF2
import docx
from io import BytesIO

from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import YouTubeSearchTool

from langgraph.graph import StateGraph, END,START
from langgraph.graph.message import add_messages
from Agentswithtools import agent_executor


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "add_messages"]
    user_input: str 
def should_continue(state: AgentState) -> Literal["continue","end"]:
    if not state["messages"]: return "end"
    last_message = state["messages"][-1]
    if getattr(last_message,"tool_calls", None):
        return "continue"
    return "end"

def call_agent_node(state: AgentState) -> AgentState:

    result = agent_executor.invoke({"input": state["user_input"]})
    state["messages"].append(result)
    return state

def action_node(state: AgentState) -> AgentState:
    state["messages"].append("Action node executed")
    return state

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("agent", call_agent_node)
    graph.add_node("action", action_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"continue": "action", "end": END}
    )
    graph.add_edge("action", "agent")
    return graph.compile()
