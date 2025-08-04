import json
import re
from typing import List, Dict, Optional, Any
from PIL import Image

# Put your Gemini API setup here
from config import GEMINI_API_KEY, GEMINI_MODEL_NAME
from history import ActionHistory
import google.generativeai as genai

genai.configure(api_key=GEMINI_API_KEY)


def clean_json_response(response_text: str) -> str:
    """Clean and extract JSON response"""
    # Remove markdown code blocks
    response_text = re.sub(r'```json\s*', '', response_text)
    response_text = re.sub(r'```\s*', '', response_text)
    
    # Remove leading and trailing whitespace
    response_text = response_text.strip()
    
    # Find JSON array
    json_pattern = r'\[\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\s*\]'
    json_match = re.search(json_pattern, response_text, re.DOTALL)
    
    if json_match:
        return json_match.group()
    
    # If no proper JSON array is found, try to extract content between []
    start_idx = response_text.find('[')
    end_idx = response_text.rfind(']')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return response_text[start_idx:end_idx + 1]
    
    return response_text


def generate_thinking_step(
    task_description: str,
    subtask_description: str,
    screenshot_path: str,
    future_tasks: Optional[List[str]] = None,
    action_history: Optional[ActionHistory] = None,
    current_os: str = "Windows"
) -> str:
    """
    First step: Generate thinking and analysis
    Returns raw thinking text instead of JSON
    """
    
    if future_tasks is None:
        future_tasks = []
    if action_history is None:
        action_history = ActionHistory()
    
    # Get key information from the last action
    last_action_info = action_history.get_last_action_summary()
    
    thinking_prompt = """
    You are an expert in graphical user interfaces and Python code. You are responsible for executing the current subtask: `{subtask_description}` of the larger goal: `{task_description}`.    
    First, you need to follow this enhanced three-step reasoning process:

    1. **Step 1 Analyze**: Carefully observe the current screenshot and decide if the current subtask `{subtask_description}` is already completed.
    - If it is completed, immediately return `agent.done()`, do not continue.
    2. **Step 2 Reflect**: Analyze the action history to detect if you're stuck in an execution loop:
    - Look for repeated similar actions or alternating patterns
    - If you detect a loop, encourage the computer agent to try a new action.
    3. **Step 3 Verify**: Compare the {last_action} in history with current screen to determine if the previous action was successful:
    - Check if the screen state changed as expected
    - If the previous action was not successful, provide a reason for the failure.

    **The future subtasks {future_tasks} will be done in the future by me. You must only perform the current subtask: `{subtask_description}`. Do not try to do future subtasks.**

    You are working in {current_os}. You must only complete the subtask provided and not the larger goal.

    You are provided with:
    1. A screenshot of the current time step
    2. (Optional) The history of your previous interactions with the UI:
    {action_history}

    Please provide your detailed analysis and reasoning for each of the three steps above. Write in clear, natural language - no JSON formatting required. This analysis will be used by the next step to generate the final action.
    """
    
    formatted_prompt = thinking_prompt.format(
        task_description=task_description,
        subtask_description=subtask_description,
        future_tasks=json.dumps(future_tasks),
        action_history=action_history.get_history_string(),
        last_action=last_action_info,
        current_os=current_os
    )

    # Load screenshot
    image = Image.open(screenshot_path)
    
    # Call Gemini API for thinking step
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    response = model.generate_content([formatted_prompt, image])
    
    if not response or not response.text:
        return None
    
    # Return raw thinking text
    return response.text.strip()


def generate_runner_action(
    task_description: str,
    subtask_description: str,
    screenshot_path: str,
    done_tasks: Optional[List[str]] = None,
    future_tasks: Optional[List[str]] = None,
    action_history: Optional[ActionHistory] = None,
    current_os: str = "Windows"
) -> Dict[str, Any]:
    """
    Generate next action step
    
    Args:
        task_description: Overall task description
        subtask_description: Current subtask description
        screenshot_path: Current screenshot path
        done_tasks: List of completed tasks
        future_tasks: List of future tasks to execute
        action_history: Action history record
        current_os: Current operating system
        
    Returns:
        Dictionary containing action information
    """
    
    if done_tasks is None:
        done_tasks = []
    if future_tasks is None:
        future_tasks = []
    if action_history is None:
        action_history = ActionHistory()
        print("WARNING: No action_history provided to runner, created new instance")
    else:
        print(f"DEBUG: Received action_history with {action_history.get_action_count()} existing actions")
    
    # First step: Get thinking results
    print("DEBUG: Starting first LLM call for thinking step...")
    thinking_result = generate_thinking_step(
        task_description=task_description,
        subtask_description=subtask_description,
        screenshot_path=screenshot_path,
        future_tasks=future_tasks,
        action_history=action_history,
        current_os=current_os
    )
    
    if not thinking_result:
        print("ERROR: Failed to get thinking results from first LLM call")
        return None
    
    print(f"DEBUG: Got thinking results: {thinking_result}")
    print("DEBUG: Starting second LLM call for action generation...")
    
    
    # Build prompt template for second step - action generation
    action_prompt = """
    You are an expert in graphical user interfaces and Python code. You are responsible for executing the current subtask: `{subtask_description}` of the larger goal: `{task_description}`.    

    {json_error_context}

    ## Important Instructions

    **The future subtasks {future_tasks} will be done in the future by me. You must only perform the current subtask: `{subtask_description}`. Do not try to do future subtasks.**

    You are working in {current_os}. You must only complete the subtask provided and not the larger goal.

    ## Previous Analysis Results
    
    Here is the detailed analysis from the previous thinking step:
    
    {thinking_analysis}

    ## Available Tools

    You have access to the following class and methods to interact with the UI:

    ### Agent Class Methods

    ```python
    class Agent:

        def click(element_description: str, num_clicks: int = 1, button_type: str = 'left', hold_keys: List = []):
            '''Click on the element
            Args:
                element_description: str, a detailed descriptions of which element to click on. 
                                This description should be at least a full sentence.
                num_clicks: int, number of times to click the element
                button_type: str, which mouse button to press can be "left", "middle", or "right"
                hold_keys: List, list of keys to hold while clicking
            '''

        def switch_applications(app_code):
            '''Switch to a different application that is already open
            Args:
                app_code: str, the code name of the application to switch to from the provided list of open applications
            '''

        def open(app_or_filename: str):
            '''Open any application or file with name app_or_filename. 
            Use this action to open applications or files on the desktop, do not open manually.
            Args:
                app_or_filename: str, the name of the application or filename to open
            '''

        def type(element_description: Optional[str] = None, text: str = '', overwrite: bool = False, enter: bool = False):
            '''Type text into a specific element
            Args:
                element_description: str, a detailed description of which element to enter text in. 
                                This description should be at least a full sentence.
                text: str, the text to type
                overwrite: bool, Assign it to True if the text should overwrite the existing text, 
                            otherwise assign it to False. Using this argument clears all text in an element.
                enter: bool, Assign it to True if the enter key should be pressed after typing the text, 
                            otherwise assign it to False.
            '''

        def drag_and_drop(starting_description: str, ending_description: str, hold_keys: List = []):
            '''Drag from the starting description to the ending description
            Args:
                starting_description: str, a very detailed description of where to start the drag action. 
                                    This description should be at least a full sentence.
                ending_description: str, a very detailed description of where to end the drag action. 
                                This description should be at least a full sentence.
                hold_keys: List, list of keys to hold while dragging
            '''

        def highlight_text_span(starting_phrase: str, ending_phrase: str):
            '''Highlight a text span between a provided starting phrase and ending phrase. 
            Use this to highlight words, lines, and paragraphs.
            Args:
                starting_phrase: str, the phrase that denotes the start of the text span you want to highlight. 
                            If you only want to highlight one word, just pass in that single word.
                ending_phrase: str, the phrase that denotes the end of the text span you want to highlight. 
                            If you only want to highlight one word, just pass in that single word.
            '''

        def set_cell_values(cell_values: Dict[str, Any], app_name: str, sheet_name: str):
            '''Use this to set individual cell values in a spreadsheet. 
            For example, setting A2 to "hello" would be done by passing {{"A2": "hello"}} as cell_values. 
            The sheet must be opened before this command can be used.
            Args:
                cell_values: Dict[str, Any], A dictionary of cell values to set in the spreadsheet. 
                            The keys are the cell coordinates in the format "A1", "B2", etc.
                            Supported value types include: float, int, string, bool, formulas.
                app_name: str, The name of the spreadsheet application. For example, "Some_sheet.xlsx".
                sheet_name: str, The name of the sheet in the spreadsheet. For example, "Sheet1".
            '''

        def scroll(element_description: str, clicks: int, shift: bool = False):
            '''Scroll the element in the specified direction
            Args:
                element_description: str, a very detailed description of which element to enter scroll in. 
                                This description should be at least a full sentence.
                clicks: int, the number of clicks to scroll can be positive (up) or negative (down).
                shift: bool, whether to use shift+scroll for horizontal scrolling
            '''

        def hotkey(keys: List):
            '''Press a hotkey combination
            Args:
                keys: List, the keys to press in combination in a list format (e.g. ['ctrl', 'c'])
            '''

        def hold_and_press(hold_keys: List, press_keys: List):
            '''Hold a list of keys and press a list of keys
            Args:
                hold_keys: List, list of keys to hold
                press_keys: List, list of keys to press in a sequence
            '''

        def wait(time: float):
            '''Wait for a specified amount of time
            Args:
                time: float, the amount of time to wait in seconds
            '''

        def done(return_value: Optional[Union[Dict, str, List, Tuple, int, float, bool]] = None):
            '''End the current task with a success and the required return value'''

        def fail():
            '''End the current task with a failure, and replan the whole task.'''
    ```    
    ## Response Format
    **CRITICAL: You must output ONLY a valid JSON array. The format is very strict:**

    Based on the previous analysis, organize your response into the following JSON format:

    [
    {{
        "screen_analysis": "Extract and summarize the screen analysis from the previous thinking step. Focus on the current subtask `{subtask_description}` and note any changes from the previous action. Describe the visible UI elements, their states, and their relevance to the current subtask.",
        "previous_action_verification": "Extract and summarize whether the previous action was successful based on the thinking analysis. If the previous action was not successful, note here and provide a reason for the failure.",
        "next_action": "Based on the current screenshot and your thinking results about the UI, decide on the next action in natural language to accomplish current subtask.",
        "grounded_action": "Translate the next action into code using the provided API methods. Format the code like this:agent.click(\"The menu button at the top right of the window\", 1, \"left\") only write the code, do not write any other text. If you think the current subtask is already completed, return agent.done() in the code block. If you think current subtask cannot be completed, return agent.fail() in the code block. Do not do anything other than the exact specified task. Whenever possible, your grounded action should use hot-keys with the agent.hotkey() action instead of clicking or dragging."
    }}
    ]

    Only output this json. Do not wrap it in code blocks or add any other text.    

    ##You must strictly follow these Execution Rules
    - Always perform loop detection and last action analysis first
    - If you detect an execution loop, you MUST use a different approach or method
    - Carefully observe and understand the current state of the computer before generating the action
    - Only perform one action at a time
    - Do not put anything other than python code in "grounded_action"
    - You can only use one function call at a time
    - Do not put more than one function call in "grounded_action"
    - You must use only the available methods provided above to interact with the UI, do not invent new methods
    - Only return one code block every time
    - There must be a single line of code in "grounded_action" code block
    - you'd better use positional parameters instead of named parameters in the grounded_action code block
    - If you think the task is already completed, return `agent.done()` in the code block
    - If you think the task cannot be completed, return `agent.fail()` in the code block
    - Do not do anything other than the exact specified task
    - Return with `agent.done()` immediately after the task is completed or `agent.fail()` if it cannot be completed
    - *very important*:Whenever possible, your grounded action should use hot-keys with the `agent.hotkey()` action instead of clicking or dragging
    - *very important*:Whenever possible, your grounded action should use open() to open applications or files instead of clicking on them
    """

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Build different prompt content based on attempt number
            json_error_context = ""
            if attempt > 0:
                json_error_context = f"""
            **IMPORTANT - RETRY #{attempt}**: The previous response had JSON formatting errors. 
            Please ensure your output is a valid JSON array with proper syntax:
            - Each field must be properly quoted strings
            - All objects must be properly closed with }}
            - The array must be properly closed with ]
            - No trailing commas
            - No extra text outside the JSON array
            """
            
            # Format prompt for second step
            formatted_prompt = action_prompt.format(
                task_description=task_description,
                subtask_description=subtask_description,
                future_tasks=json.dumps(future_tasks),
                current_os=current_os,
                json_error_context=json_error_context,
                thinking_analysis=thinking_result
            )

            # Load screenshot
            image = Image.open(screenshot_path)
            print(f"DEBUG: Loaded screenshot from {screenshot_path} ")
            
            # Call Gemini API for second step
            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            response = model.generate_content([formatted_prompt, image])

            if not response or not response.text:
                print(f"[RETRY {attempt + 1}/{max_retries}] Empty response from Gemini")
                if attempt == max_retries - 1:
                    return {}
                continue

            # Clean and parse JSON response
            response_text = response.text.strip()
            print(f"[RETRY {attempt + 1}/{max_retries}] Raw response: {response_text[:200]}...")
            
            cleaned_json = clean_json_response(response_text)
            print(f"[RETRY {attempt + 1}/{max_retries}] Cleaned JSON: {cleaned_json}")
            
            # Try to parse JSON
            try:
                action_data = json.loads(cleaned_json)
            except json.JSONDecodeError as e:
                print(f"[RETRY {attempt + 1}/{max_retries}] JSON parsing error: {e}")
                print(f"[RETRY {attempt + 1}/{max_retries}] Problematic text: {cleaned_json}")
                
                if attempt == max_retries - 1:
                    print("Max retries reached for JSON parsing. Runner failed.")
                    return None
                else:
                    print(f"Retrying... ({attempt + 2}/{max_retries})")
                    continue
            
            # Validate response structure
            if not isinstance(action_data, list) or len(action_data) == 0:
                print(f"[RETRY {attempt + 1}/{max_retries}] Invalid action data structure")
                if attempt == max_retries - 1:
                    print("Max retries reached for action validation. Runner failed.")
                    return None
                continue
                
            action_result = action_data[0]
            
            # Validate required fields
            required_fields = ["screen_analysis", "previous_action_verification", "next_action", "grounded_action"]
            for field in required_fields:
                if field not in action_result:
                    print(f"Warning: Missing field {field}")
                    action_result[field] = "Not provided"
            
            print(f"[SUCCESS] Generated runner action with {len(required_fields)} fields")
            return action_result
            
        except Exception as e:
            print(f"[RETRY {attempt + 1}/{max_retries}] Unexpected error: {e}")
            if 'response' in locals() and hasattr(response, 'text'):
                print(f"Raw response: {response.text}")
            
            if attempt == max_retries - 1:
                print("Max retries reached due to unexpected error. Runner failed.")
                return None
    
    print("All retry attempts completed without success. Runner failed.")
    return None


def execute_runner_step(
    task_description: str,
    subtask: Dict[str, Any],
    screenshot_path: str,
    future_tasks: Optional[List[str]] = None,
    action_history: Optional[ActionHistory] = None
) -> Dict[str, Any]:
    """
    Execute one step of the runner
    
    Args:
        task_description: Overall task description
        subtask: Current subtask (contains 'subtask' and 'description' fields)
        screenshot_path: Screenshot path
        done_tasks: List of completed tasks
        future_tasks: List of future tasks
        action_history: Action history
        
    Returns:
        Generated action data
    """
    
    subtask_description = subtask.get('description', str(subtask))
    
    return generate_runner_action(
        task_description=task_description,
        subtask_description=subtask_description,
        screenshot_path=screenshot_path,
        future_tasks=future_tasks,
        action_history=action_history
    )