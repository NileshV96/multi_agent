# main.py

# main.py

from openai import OpenAI
import json
import os
from dotenv import load_dotenv

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in .env file")

# -----------------------------
# Initialize OpenAI Client
# -----------------------------
client = OpenAI(api_key=api_key)

# -----------------------------
# System Prompt (Agent Brain)
# -----------------------------
SYSTEM_PROMPT = """
You are an intelligent AI agent.

Your job is to:
1. Understand the user query
2. Identify the intent
3. Break it into tasks
4. Provide final answer

IMPORTANT:
- Always return output in JSON format
- Do NOT return plain text
- Follow this schema strictly:

{
  "intent": "string",
  "tasks": ["list of steps"],
  "final_answer": "string"
}
"""

# -----------------------------
# Validation Function
# -----------------------------
def validate_output(output: dict) -> bool:
    required_keys = ["intent", "tasks", "final_answer"]

    for key in required_keys:
        if key not in output:
            return False

    if not isinstance(output["tasks"], list):
        return False

    return True

# -----------------------------
# Agent Function
# -----------------------------
def run_agent(user_query: str) -> dict:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query}
            ],
            temperature=0,
            response_format={"type": "json_object"}  # 🔥 Enforces valid JSON
        )

        output = response.choices[0].message.content

        # Debug log (very useful)
        print("\n🔍 Raw LLM Output:")
        print(output)

        parsed_output = json.loads(output)

        # Validate structure
        if validate_output(parsed_output):
            return parsed_output
        else:
            return {
                "error": "Invalid structure",
                "data": parsed_output
            }

    except json.JSONDecodeError:
        return {
            "error": "Invalid JSON format",
            "raw_output": output
        }

    except Exception as e:
        return {
            "error": str(e)
        }

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    user_query = input("Enter your query: ")

    result = run_agent(user_query)

    print("\n✅ Final Output:")
    print(json.dumps(result, indent=2))