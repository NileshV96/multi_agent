# =========================================
# Multi-Agent MCP System (Multi-Tool)
# =========================================

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
# TOOLS (MCP Style)
# =========================================

def get_weather(latitude: float, longitude: float):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
    return requests.get(url).json()


def get_ip_info():
    url = "http://ip-api.com/json/"
    return requests.get(url).json()


def search_duckduckgo(query: str):
    url = f"https://api.duckduckgo.com/?q={query}&format=json"
    return requests.get(url).json()


# =========================================
# TOOL REGISTRY (IMPORTANT)
# =========================================
TOOLS = {
    "get_weather": get_weather,
    "get_ip_info": get_ip_info,
    "search": search_duckduckgo
}


# =========================================
# Planner Agent
# =========================================
def planner_agent(user_query: str):

    PROMPT = """
    You are a Planner Agent.

    Return JSON:
    {
      "intent": "string",
      "tasks": ["steps"]
    }
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": user_query}
        ],
        response_format={"type": "json_object"}
    )

    output = response.choices[0].message.content
    print("\n🧠 Planner:", output)

    return json.loads(output)


# =========================================
# Executor Agent (Tool Selector)
# =========================================
def executor_agent(planner_output: dict, user_query: str):

    PROMPT = f"""
    You are a Tool Selection Agent.

    Available Tools:
    1. get_weather(latitude, longitude)
    2. get_ip_info()
    3. search(query)

    Decide:
    - which tool to use
    - parameters

    Return JSON:
    {{
      "use_tool": true/false,
      "tool_name": "string",
      "tool_params": {{}}
    }}

    Query: {user_query}
    Plan: {planner_output}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": PROMPT}],
        response_format={"type": "json_object"}
    )

    output = response.choices[0].message.content
    print("\n⚙️ Executor:", output)

    result = json.loads(output)

    # Execute tool dynamically
    if result.get("use_tool"):
        tool_name = result.get("tool_name")
        params = result.get("tool_params", {})

        if tool_name in TOOLS:
            try:
                tool_result = TOOLS[tool_name](**params)
                result["tool_result"] = tool_result
            except Exception as e:
                result["tool_result"] = {"error": str(e)}

    return result


# =========================================
# Responder Agent
# =========================================
def responder_agent(planner, executor):

    PROMPT = f"""
    You are a Responder Agent.

    Planner:
    {planner}

    Executor:
    {executor}

    Give final answer.
    Return JSON:
    {{
      "final_answer": "string"
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": PROMPT}],
        response_format={"type": "json_object"}
    )

    output = response.choices[0].message.content
    print("\n💬 Responder:", output)

    return json.loads(output)


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

app = graph.compile()


# =========================================
# Run
# =========================================
def run(query: str):
    state = {
        "user_query": query,
        "planner": {},
        "executor": {},
        "final": {}
    }

    result = app.invoke(state)
    return result["final"]


# =========================================
# Main
# =========================================
if __name__ == "__main__":
    q = input("Enter query: ")
    result = run(q)
    print("\n✅ Final Output:")
    print(json.dumps(result, indent=2))