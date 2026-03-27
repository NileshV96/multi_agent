# =========================================
# FastAPI + Multi-Agent MCP System
# =========================================

# =========================================
# Imports
# =========================================
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import json
import os
import requests
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph


# =========================================
# Load Environment Variables
# =========================================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set")


# =========================================
# Initialize OpenAI Client
# =========================================
client = OpenAI(api_key=api_key)


# =========================================
# FastAPI App Initialization
# =========================================
app = FastAPI(title="Multi-Agent MCP API")


# =========================================
# Request Schema
# =========================================
class QueryRequest(BaseModel):
    query: str


# =========================================
# TOOLS
# =========================================
def get_weather(latitude: float, longitude: float):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
    return requests.get(url).json()


def get_ip_info():
    return requests.get("http://ip-api.com/json/").json()


def search_duckduckgo(query: str):
    return requests.get(f"https://api.duckduckgo.com/?q={query}&format=json").json()


TOOLS = {
    "get_weather": get_weather,
    "get_ip_info": get_ip_info,
    "search": search_duckduckgo
}


# =========================================
# AGENTS
# =========================================
def planner_agent(user_query: str):
    PROMPT = """
    Return JSON:
    {
      "intent": "string",
      "tasks": ["steps"]
    }
    """

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": user_query}
        ],
        response_format={"type": "json_object"}
    )

    return json.loads(res.choices[0].message.content)


def executor_agent(planner_output, user_query):
    PROMPT = f"""
    Available Tools:
    get_weather(latitude, longitude)
    get_ip_info()
    search(query)

    Return JSON:
    {{
      "use_tool": true/false,
      "tool_name": "string",
      "tool_params": {{}}
    }}

    Query: {user_query}
    Plan: {planner_output}
    """

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": PROMPT}],
        response_format={"type": "json_object"}
    )

    result = json.loads(res.choices[0].message.content)

    if result.get("use_tool"):
        tool_name = result.get("tool_name")
        params = result.get("tool_params", {})

        if tool_name in TOOLS:
            try:
                result["tool_result"] = TOOLS[tool_name](**params)
            except Exception as e:
                result["tool_result"] = {"error": str(e)}

    return result


def responder_agent(planner, executor):
    PROMPT = f"""
    Planner: {planner}
    Executor: {executor}

    Return JSON:
    {{
      "final_answer": "string"
    }}
    """

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": PROMPT}],
        response_format={"type": "json_object"}
    )

    return json.loads(res.choices[0].message.content)


# =========================================
# LangGraph State
# =========================================
class AgentState(TypedDict):
    user_query: str
    planner: dict
    executor: dict
    final: dict


# =========================================
# Nodes
# =========================================
def planner_node(state: AgentState):
    state["planner"] = planner_agent(state["user_query"])
    return state


def executor_node(state: AgentState):
    state["executor"] = executor_agent(state["planner"], state["user_query"])
    return state


def responder_node(state: AgentState):
    state["final"] = responder_agent(state["planner"], state["executor"])
    return state


# =========================================
# Graph
# =========================================
graph = StateGraph(AgentState)

graph.add_node("planner", planner_node)
graph.add_node("executor", executor_node)
graph.add_node("responder", responder_node)

graph.set_entry_point("planner")
graph.add_edge("planner", "executor")
graph.add_edge("executor", "responder")

workflow = graph.compile()


# =========================================
# API Endpoint
# =========================================
@app.post("/query")
def query_agent(request: QueryRequest):
    state = {
        "user_query": request.query,
        "planner": {},
        "executor": {},
        "final": {}
    }

    result = workflow.invoke(state)

    return result["final"]


# =========================================
# Health Check
# =========================================
@app.get("/")
def health():
    return {"status": "running"}