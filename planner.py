# planner/planner.py
import json
from typing import List, Dict, Optional
from PIL import Image
import re

# Gemini API setup
from config import GEMINI_API_KEY, GEMINI_MODEL_NAME
import google.generativeai as genai

# RAG functionality
from utils.rag_retriever import get_knowledge_context, is_knowledge_base_available

# RAG configuration
ENABLE_RAG = False
_last_user_command = ""
_last_knowledge_context = ""

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL_NAME)

PLANNER_PROMPT_TEMPLATE = """
You are an expert planning agent specialized in solving GUI-based navigation and automation tasks on desktop environments.

Your job is to generate or revise a **step-by-step plan** for completing a user-requested task, based on the current desktop screenshot and task context.

you must follow this enhanced four-step reasoning process:
1. **Step 1 observation**:Carefully observe the current screenshot
2. **Step 2 analysis**: Analyze the user command and current desktop state
3. **Step 3 planning**: Generate a detailed step-by-step plan to complete the task

You are provided with:
1. A high-level **user task description**: "{user_command}"
2. The **current screenshot of the desktop** (image provided)
3. (Optional) A list of **remaining subtasks from previous planning**: {future_steps}
4. (Optional) A list of **compeleted subtasks**: {completed_steps}
5. (Optional) A list of **failed subtasks**: {failed_steps}
6. (Optional) Relevant **knowledge base context**: {knowledge_context}(but may not be useful)
{json_error_context}

---

Your responsibilities:
1. Generate a new plan or revise the pre-existing plan to complete the task
2. Ensure the plan is concise and contains only necessary steps
3. Carefully observe and understand the current state of the computer before generating your plan
4. Avoid including steps in your plan that the task does not ask for
5. **Important**: Consider the failed subtasks and find alternative approaches to achieve the same goals

Below are important considerations when generating your plan:
##You must strictly follow these Execution Rules
1. Provide the plan in a step-by-step format with detailed descriptions for each subtask.
2. Do not repeat subtasks that have already been successfully completed.
3. Do not repeat subtasks that have failed - instead, find alternative approaches.
4. Do not include verification steps in your planning.
5. Do not include optional steps in your planning.
6. Do not include unnecessary steps in your planning.
7. Do not specify the exact GUI elements interaction details like "double click...", focus on the high-level plan.
8. If you need to set a cell to x, do not split the task and directly generate a subplan "set cell to x". 
9. If you think the current task has been completed, please output a special completion marker shown below.
10. **For failed subtasks**: Analyze why they failed and create alternative subtasks that avoid the same failure points.

---

## Context Information:

**Remaining Subtasks**: {future_steps}  
**compeleted subtasks**: {completed_steps}
**Failed Subtasks**: {failed_steps}

If there are failed subtasks, please analyze the failure reasons and create alternative approaches to achieve the same goals.

---

**CRITICAL: You must output ONLY one of these two formats:**

**Format 1: If the task is NOT completed, output a JSON list of subtasks:**
[
  {{"subtask": 1, "description": "Detailed information about executing this step"}},
  {{"subtask": 2, "description": "Detailed information about executing this step"}},
  {{"subtask": 3, "description": "Detailed information about executing this step"}}
]

**Format 2: If the task IS completed, output exactly this completion marker:**
{{"task_completed": true, "completion_message": "Task has been successfully completed"}}

Only output one of these formats. Do not wrap in code blocks or add any other text.
"""  # Keep your original prompt

def get_rag_knowledge_if_needed(user_command: str) -> str:
    """
    Fetch RAG knowledge only when the user command changes.

    Args:
        user_command: The user task description.

    Returns:
        Retrieved knowledge context string.
    """
    global _last_user_command, _last_knowledge_context

    if not ENABLE_RAG:
        return ""

    if not is_knowledge_base_available():
        return ""

    if user_command == _last_user_command:
        if _last_knowledge_context:
            print("Using cached knowledge context")
        return _last_knowledge_context

    print("User command changed. Fetching new knowledge context...")
    try:
        knowledge_context = get_knowledge_context(user_command)
        if knowledge_context:
            print(f"Retrieved knowledge context ({len(knowledge_context)} characters)")
        else:
            print("No related knowledge context found")
        _last_user_command = user_command
        _last_knowledge_context = knowledge_context or ""
        return _last_knowledge_context
    except Exception as e:
        print(f"Knowledge base retrieval failed: {e}")
        return ""

def clear_rag_cache():
    """Clear RAG cache."""
    global _last_user_command, _last_knowledge_context
    _last_user_command = ""
    _last_knowledge_context = ""
    print("RAG cache cleared")

def set_rag_enabled(enabled: bool):
    """Enable or disable RAG."""
    global ENABLE_RAG
    ENABLE_RAG = enabled
    if enabled:
        print("RAG enabled")
    else:
        print("RAG disabled")
        clear_rag_cache()

def clean_json_response(response_text: str) -> str:
    """
    Clean and extract JSON from LLM output text.

    Returns:
        A string that is parseable by json.loads, or the original text.
    """
    # Remove markdown formatting
    response_text = re.sub(r'```json\s*', '', response_text)
    response_text = re.sub(r'```\s*', '', response_text)

    response_text = response_text.strip()

    # Try direct parse first
    try:
        json.loads(response_text)
        return response_text
    except json.JSONDecodeError:
        pass

    # Check for completion marker
    completion_pattern = r'\{\s*"task_completed"\s*:\s*true\s*,\s*"completion_message"\s*:\s*"[^"]*"\s*\}'
    match = re.search(completion_pattern, response_text, re.DOTALL)
    if match:
        return match.group()

    # Try array extraction
    array_pattern = r'\[\s*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\s*,?\s*)*\]'
    match = re.search(array_pattern, response_text, re.DOTALL)
    if match:
        return match.group()

    # Fallback bracket extraction
    start = response_text.find('[')
    end = response_text.rfind(']')
    if start != -1 and end != -1 and end > start:
        return response_text[start:end + 1]

    start = response_text.find('{')
    end = response_text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return response_text[start:end + 1]

    return response_text

def generate_plan(
    user_command: str,
    screenshot_path: str,
    future_steps: Optional[List[Dict]] = None,
    failed_steps: Optional[List[Dict]] = None,
    completed_steps: Optional[List[Dict]] = None
) -> List[Dict]:

    knowledge_context = get_rag_knowledge_if_needed(user_command)

    if not ENABLE_RAG:
        print("RAG is disabled")
    elif not knowledge_context:
        print("No knowledge context used")
    else:
        print(f"Using knowledge context ({len(knowledge_context)} characters)")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            json_error_context = ""
            if attempt > 0:
                json_error_context = f"""
6. **IMPORTANT - RETRY #{attempt}**: The previous response had JSON formatting errors. 
   Please ensure your output is a valid JSON array with proper syntax:
   - Each object must end with }}
   - The array must end with ]
   - No trailing commas
   - No extra text outside the JSON array
"""

            prompt = PLANNER_PROMPT_TEMPLATE.format(
                knowledge_context=knowledge_context,
                user_command=user_command,
                future_steps=json.dumps(future_steps or []),
                completed_steps=json.dumps(completed_steps or []),
                failed_steps=json.dumps(failed_steps or []),
                json_error_context=json_error_context
            )

            image = Image.open(screenshot_path)
            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            response = model.generate_content([prompt, image])

            if not response or not response.text:
                print(f"[RETRY {attempt + 1}/{max_retries}] Empty response from Gemini")
                if attempt == max_retries - 1:
                    return []
                continue

            response_text = response.text.strip()
            print(f"[RETRY {attempt + 1}/{max_retries}] Raw response: {response_text[:200]}...")

            cleaned_json = clean_json_response(response_text)
            print(f"[RETRY {attempt + 1}/{max_retries}] Cleaned JSON: {cleaned_json}")

            try:
                parsed_response = json.loads(cleaned_json)
            except json.JSONDecodeError as e:
                print(f"[RETRY {attempt + 1}/{max_retries}] JSON parsing error: {e}")
                print(f"[RETRY {attempt + 1}/{max_retries}] Problematic text: {cleaned_json}")
                if attempt == max_retries - 1:
                    print("Max retries reached for JSON parsing. Planner failed.")
                    return None
                else:
                    print(f"Retrying... ({attempt + 2}/{max_retries})")
                    continue

            if isinstance(parsed_response, dict) and parsed_response.get("task_completed") is True:
                print(f"[SUCCESS] Task completion detected: {parsed_response.get('completion_message', 'No message')}")
                return {
                    "task_completed": True,
                    "completion_message": parsed_response.get("completion_message", "Task completed")
                }

            plan = parsed_response
            if not isinstance(plan, list):
                print(f"[RETRY {attempt + 1}/{max_retries}] Invalid plan data structure - not a list")
                if attempt == max_retries - 1:
                    print("Max retries reached for plan validation. Planner failed.")
                    return None
                continue

            validated_plan = []
            for i, step in enumerate(plan):
                if not isinstance(step, dict):
                    print(f"Warning: Step {i} is not a dictionary, skipping")
                    continue

                if 'subtask' not in step:
                    step['subtask'] = i + 1
                    print(f"Warning: Missing subtask field in step {i}, assigned {step['subtask']}")

                if 'description' not in step:
                    step['description'] = "No description provided"
                    print(f"Warning: Missing description field in step {i}")

                validated_plan.append(step)

            print(f"[SUCCESS] Parsed plan with {len(validated_plan)} steps")
            return validated_plan

        except Exception as e:
            print(f"[RETRY {attempt + 1}/{max_retries}] Unexpected error: {e}")
            if 'response' in locals() and hasattr(response, 'text'):
                print(f"Raw response: {response.text}")

            if attempt == max_retries - 1:
                print("Max retries reached due to unexpected error. Planner failed.")
                return None

    print("All retry attempts completed without success. Planner failed.")
    return None

# RAG utility functions
def enable_rag():
    """Enable RAG functionality."""
    set_rag_enabled(True)

def disable_rag():
    """Disable RAG functionality."""
    set_rag_enabled(False)

def is_rag_enabled() -> bool:
    """Check whether RAG is enabled."""
    return ENABLE_RAG

def get_rag_status():
    """Print current RAG status."""
    status = {
        "enabled": ENABLE_RAG,
        "knowledge_base_available": is_knowledge_base_available() if ENABLE_RAG else False,
        "has_cached_knowledge": bool(_last_knowledge_context)
    }

    print("RAG Status:")
    print(f"  - Enabled: {'Yes' if status['enabled'] else 'No'}")
    print(f"  - Knowledge Base Available: {'Yes' if status['knowledge_base_available'] else 'No'}")
    print(f"  - Has Cached Knowledge: {'Yes' if status['has_cached_knowledge'] else 'No'}")

    return status