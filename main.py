# =========================================
# Multi-Agent System with Conditional Routing (LangGraph)
# =========================================
# Agents:
# 1. Planner Agent
# 2. Executor Agent
# 3. Responder Agent
# =========================================


# =========================================
# Imports
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
    raise ValueError("OPENAI_API_KEY is not set in .env file")


# =========================================
# Initialize OpenAI Client
# =========================================
client = OpenAI(api_key=api_key)


# =========================================
# Tool: Get Weather Data
# =========================================
def get_weather(latitude: float, longitude: float) -> dict:
    """
    Fetch current weather data using Open-Meteo API.
    """
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"

    try:
        response = requests.get(url)

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": "Failed to fetch weather data"}

    except Exception as e:
        return {"error": str(e)}


# =========================================
# Planner Agent
# =========================================
def planner_agent(user_query: str) -> dict:
    """
    Planner Agent:
    - Understands user query
    - Extracts intent
    - Breaks into tasks
    """

    PLANNER_PROMPT = """
    You are a Planner Agent.

    Your job:
    1. Understand user query
    2. Identify intent
    3. Break into tasks

    Return JSON:

    {
      "intent": "string",
      "tasks": ["list of steps"]
    }
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": user_query}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    output = response.choices[0].message.content

    print("\n🧠 Planner Output:")
    print(output)

    return json.loads(output)


# =========================================
# Executor Agent
# =========================================
def executor_agent(planner_output: dict) -> dict:
    """
    Executor Agent:
    - Decides tool usage
    - Executes tool if needed
    """

    EXECUTOR_PROMPT = f"""
    You are an Executor Agent.

    Based on this plan:
    {planner_output}

    Decide:
    - Whether a tool is needed
    - Which tool to call

    Available tool:
    get_weather(latitude, longitude)

    Return JSON:

    {{
      "use_tool": true/false,
      "tool_name": "string or null",
      "tool_params": {{}}
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": EXECUTOR_PROMPT}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    output = response.choices[0].message.content

    print("\n⚙️ Executor Output:")
    print(output)

    result = json.loads(output)

    # ====================================
    # Tool Execution Logic
    # ====================================
    if result.get("use_tool") and result.get("tool_name") == "get_weather":
        params = result.get("tool_params", {})
        tool_result = get_weather(**params)
        result["tool_result"] = tool_result

    return result


# =========================================
# Responder Agent
# =========================================
def responder_agent(planner_output: dict, executor_output: dict) -> dict:
    """
    Responder Agent:
    - Generates final answer
    """

    RESPONDER_PROMPT = f"""
    You are a Responder Agent.

    Planner Output:
    {planner_output}

    Executor Output:
    {executor_output}

    Generate a clear and helpful final answer.

    Return JSON:

    {{
      "final_answer": "string"
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": RESPONDER_PROMPT}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    output = response.choices[0].message.content

    print("\n💬 Responder Output:")
    print(output)

    return json.loads(output)


# =========================================
# LangGraph State Definition (Updated)
# =========================================
class AgentState(TypedDict):
    user_query: str
    planner_output: dict
    executor_output: dict
    final_output: dict
    use_tool: bool


# =========================================
# Planner Node
# =========================================
def planner_node(state: AgentState) -> AgentState:
    state["planner_output"] = planner_agent(state["user_query"])
    return state


# =========================================
# Executor Node (Sets use_tool flag)
# =========================================
def executor_node(state: AgentState) -> AgentState:
    executor_output = executor_agent(state["planner_output"])

    state["executor_output"] = executor_output
    state["use_tool"] = executor_output.get("use_tool", False)

    return state


# =========================================
# Responder Node
# =========================================
def responder_node(state: AgentState) -> AgentState:
    state["final_output"] = responder_agent(
        state.get("planner_output", {}),
        state.get("executor_output", {})
    )
    return state


# =========================================
# Routing Logic (Conditional Decision)
# =========================================
def route_decision(state: AgentState) -> str:
    """
    Decide next node based on tool usage
    """
    # ⚠️ Note: Executor hasn't run yet, so we simulate decision
    # For now, we trigger executor always for safety if unclear

    query = state["user_query"].lower()

    if "weather" in query:
        return "executor"
    else:
        return "responder"


# =========================================
# Build LangGraph with Conditional Routing
# =========================================
graph = StateGraph(AgentState)

graph.add_node("planner", planner_node)
graph.add_node("executor", executor_node)
graph.add_node("responder", responder_node)

graph.set_entry_point("planner")

# 🔥 Conditional Routing
graph.add_conditional_edges(
    "planner",
    route_decision,
    {
        "executor": "executor",
        "responder": "responder"
    }
)

# After executor → responder
graph.add_edge("executor", "responder")

app = graph.compile()


# =========================================
# Run LangGraph Workflow
# =========================================
def run_langgraph(user_query: str):
    initial_state = {
        "user_query": user_query,
        "planner_output": {},
        "executor_output": {},
        "final_output": {},
        "use_tool": False
    }

    result = app.invoke(initial_state)

    return result["final_output"]


# =========================================
# Main Execution Entry Point
# =========================================
if __name__ == "__main__":
    user_query = input("Enter your query: ")

    result = run_langgraph(user_query)

    print("\n✅ Final Output:")
    print(json.dumps(result, indent=2))