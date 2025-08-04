#!/usr/bin/env python3
"""
Grounding Module - Core grounding functionality
Converts runner outputs to coordinate-based actions using accessibility tree and visual parsing
"""

import ast
import json
import time
import re
import os
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
import pyautogui
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL_NAME
from utils.accessibility_tree import AccessibilityTreeExtractor, AccessibilityTreeFormatter
from utils.OmniParser.screenparser import ScreenParser
from utils.param_parser import safe_parse_params
from Agent import Agent

# Configure pyautogui safety settings
pyautogui.FAILSAFE = False  # Disable emergency stop
pyautogui.PAUSE = 0.05  # Reduced pause for better responsiveness but still safe

genai.configure(api_key=GEMINI_API_KEY)

# Global ScreenParser instance to avoid reloading models
_global_screen_parser = None
_visual_grounder_instance = None

def get_screen_parser():
    """Get global ScreenParser instance, initialize only once"""
    global _global_screen_parser
    if _global_screen_parser is None:
        try:
            print("Initializing ScreenParser (one-time setup)...")
            _global_screen_parser = ScreenParser()
            _global_screen_parser.available = True  # Mark as available
            print("ScreenParser initialized successfully")
        except Exception as e:
            print(f"Warning: ScreenParser initialization failed: {e}")
            print("Visual grounding will be disabled")
            _global_screen_parser = None
    return _global_screen_parser

def ensure_screen_parser_alive():
    """Ensure ScreenParser is alive and models are loaded"""
    global _global_screen_parser
    
    if _global_screen_parser is None:
        return get_screen_parser()
    
    # Check if ScreenParser is still alive by testing it
    try:
        # Test if models are still accessible
        if hasattr(_global_screen_parser, 'available') and _global_screen_parser.available:
            # Try a simple test to see if the parser is responsive
            # This could be a lightweight operation to verify it's working
            return _global_screen_parser
        else:
            print("ScreenParser not available, reinitializing...")
            _global_screen_parser = None
            return get_screen_parser()
    except Exception as e:
        print(f"ScreenParser appears to be dead, reinitializing: {e}")
        _global_screen_parser = None
        return get_screen_parser()


class VisualGrounding:
    """Core class for processing visual and accessibility tree grounding"""
    
    def __init__(self):
        self.model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        self.tree_extractor = AccessibilityTreeExtractor()
        self.tree_formatter = AccessibilityTreeFormatter()
        
        # Use global ScreenParser instance to avoid reloading models
        self.screen_parser = get_screen_parser()
    
    def ground_action(self, runner_result: Dict, screenshot_path: Optional[str] = None) -> str:
        """
        Core grounding function: convert runner output to grounded action with coordinates
        Logic:
        - Priority: Use accessibility tree grounding (faster and more accurate)
        - Fallback: Use ScreenParser for visual grounding
        - For all actions: click, type, scroll, drag_and_drop
        
        Args:
            runner_result: Runner output containing screen_analysis, next_action, grounded_action
            screenshot_path: Optional screenshot path
        
        Returns:
            Grounded action string with coordinates replacing descriptions
        """
        try:
            grounded_action = runner_result.get('grounded_action', 'agent.fail()')
            
            # Check if this action needs grounding
            if not self._needs_grounding(grounded_action):
                return grounded_action
            
            # Take screenshot if not provided
            if not screenshot_path:
                screenshot_path = self._take_screenshot()
            
            # Try accessibility tree grounding first
            print("Attempting accessibility tree grounding...")
            accessibility_result = self._try_accessibility_grounding(runner_result, screenshot_path)
            if accessibility_result != "agent.fail()":
                print("Accessibility tree grounding successful")
                return accessibility_result
            else:
                print("Accessibility tree grounding failed, falling back to visual grounding")
            
            # Fallback to visual grounding
            # Ensure ScreenParser is alive and ready
            self.screen_parser = ensure_screen_parser_alive()
            
            if self.screen_parser and hasattr(self.screen_parser, 'available') and self.screen_parser.available:
                print("Using visual grounding with ScreenParser...")
                visual_result = self._try_visual_grounding(runner_result, screenshot_path)
                if visual_result != "agent.fail()":
                    print("Visual grounding successful")
                    return visual_result
                else:
                    print("Visual grounding also failed")
            else:
                print("ScreenParser not available for visual grounding")
            
            # Both methods failed
            print("All grounding methods failed")
            return "agent.fail()"
            
        except Exception as e:
            print(f"Error in ground_action: {e}")
            return "agent.fail()"
    
    def _needs_grounding(self, grounded_action: str) -> bool:
        """Check if action needs grounding"""
        grounding_actions = ['agent.click(', 'agent.type(', 'agent.scroll(', 'agent.drag_and_drop(']
        return any(action in grounded_action for action in grounding_actions)
    
    def _take_screenshot(self) -> str:
        """Take screenshot and save to screenshots folder"""
        # Create screenshots directory if not exists
        screenshots_dir = "screenshots"
        os.makedirs(screenshots_dir, exist_ok=True)
        
        screenshot = pyautogui.screenshot()
        temp_path = os.path.join(screenshots_dir, "temp_screenshot.png")
        screenshot.save(temp_path)
        return temp_path
    
    def _generate_grounded_action(self, prompt: str, screenshot_path: Optional[str] = None, image_object: Optional[Image.Image] = None) -> str:
        """Generate grounded action with coordinates using LLM with retry mechanism"""
        max_retries = 3
        base_delay = 1.0  # Base delay in seconds
        
        for attempt in range(max_retries):
            try:
                # Prepare inputs
                inputs = [prompt]
                if image_object:
                    # Use provided PIL image object directly
                    inputs.append(image_object)
                elif screenshot_path:
                    try:
                        image = Image.open(screenshot_path)
                        inputs.append(image)
                    except Exception as e:
                        print(f"Warning: Could not load screenshot: {e}")
                
                # Call Gemini API
                print(f"LLM API call attempt {attempt + 1}/{max_retries}")
                response = self.model.generate_content(inputs)
                
                if not response or not response.text:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        print(f"Empty response received, retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                    else:
                        print("All retry attempts failed - empty response")
                        return "agent.fail()"
                
                # Clean response to get the grounded action code
                cleaned_response = self._clean_code_response(response.text)
                if cleaned_response != "agent.fail()":
                    if attempt > 0:
                        print(f"LLM call succeeded on attempt {attempt + 1}")
                    return cleaned_response
                else:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"LLM returned invalid response, retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                    else:
                        print("All retry attempts failed - invalid response")
                        return "agent.fail()"
                
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    print(f"LLM API error (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    print(f"All retry attempts failed. Final error: {e}")
                    return "agent.fail()"
        
        return "agent.fail()"
    
    def _clean_code_response(self, response_text: str) -> str:
        """Clean LLM response to extract the grounded action code"""
        # Remove markdown code blocks
        response_text = re.sub(r'```python\s*', '', response_text)
        response_text = re.sub(r'```\s*', '', response_text)
        
        # Remove leading/trailing whitespace
        response_text = response_text.strip()
        
        # Look for agent.* function calls
        agent_pattern = r'agent\.\w+\([^)]*\)'
        agent_matches = re.findall(agent_pattern, response_text)
        
        if agent_matches:
            # Return the first agent function call found
            return agent_matches[0]
        
        # If no agent function found, look for lines starting with agent.
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        for line in lines:
            if line.startswith('agent.'):
                return line
        
        # If nothing found, return the cleaned response or fail
        if response_text and not response_text.startswith('#'):
            return response_text
        
        return "agent.fail()"
    
    def _try_visual_grounding(self, runner_result: Dict, screenshot_path: str) -> str:
        """
        Try visual grounding using ScreenParser as fallback method
        
        Args:
            runner_result: Runner output containing action details
            screenshot_path: Screenshot path
            
        Returns:
            Grounded action with coordinates or agent.fail() if failed
        """
        try:
            # Parse screenshot with ScreenParser and get processed image
            print(f"Parsing screenshot: {screenshot_path}")
            
            # Get full parsing results including processed image
            result = self.screen_parser.parse_screenshot(screenshot_path, return_processed_image=True)
            element_info_list = []
            
            # Extract element info in the format we need
            for element in result['elements']:
                element_info = {
                    'id': element['id'],
                    'type': element['type'], 
                    'content': element['content'],
                    'interactivity': element['interactivity'],
                    'center_pixel': element['center_pixel']
                }
                element_info_list.append(element_info)
            
            if not element_info_list:
                print("No elements found by ScreenParser")
                return "agent.fail()"
            
            print(f"ScreenParser found {len(element_info_list)} elements")
            
            # Format elements for LLM (without coordinates) using ScreenParser's method
            formatted_elements = self.screen_parser.format_elements_for_llm(element_info_list)
            
            # Build grounding prompt
            grounding_prompt = self._build_visual_grounding_prompt(
                runner_result, formatted_elements
            )
            
            # Use processed image with annotations for LLM
            processed_image = result.get('labeled_image_pil')
            if processed_image:
                # Save processed image in the same folder as original with _parsed suffix
                try:
                    # Get the original image path info
                    original_dir = os.path.dirname(screenshot_path)
                    original_name = os.path.basename(screenshot_path)
                    name_without_ext = os.path.splitext(original_name)[0]
                    ext = os.path.splitext(original_name)[1]
                    
                    # Create new filename with _parsed suffix
                    parsed_filename = f"{name_without_ext}_parsed{ext}"
                    processed_image_path = os.path.join(original_dir, parsed_filename)
                    
                    processed_image.save(processed_image_path)
                    print(f"[DEBUG] Saved processed image to: {processed_image_path}")
                except Exception as e:
                    print(f"Warning: Failed to save processed image: {e}")
                
                # Call LLM with processed image using unified retry mechanism
                llm_response = self._generate_grounded_action(grounding_prompt, image_object=processed_image)
            else:
                print("Warning: No processed image available, using original")
                llm_response = self._generate_grounded_action(grounding_prompt, screenshot_path)
            
            print(f"[DEBUG] Visual LLM raw response: {llm_response}")
            
            # Check if LLM failed to generate response
            if llm_response == "agent.fail()":
                print("LLM failed to generate response for visual grounding")
                return "agent.fail()"
            
            # Validate and convert IDs to coordinates
            final_action = self._convert_ids_to_coordinates(llm_response, element_info_list)
            
            # Check if ID conversion failed
            if final_action == "agent.fail()":
                print("Visual grounding ID to coordinates conversion failed")
                return "agent.fail()"
            
            print("Visual grounding conversion successful")
            return final_action
            
        except Exception as e:
            print(f"Error in visual grounding: {e}")
            return "agent.fail()"
    
    def _build_visual_grounding_prompt(self, runner_result: Dict, formatted_elements: str) -> str:
        """Build grounding prompt for visual grounding"""
        prompt = f"""
You are a very clever UI operator and able to use mouse and keyboard to interact with the computer based on the given task and screenshot.

## Task
Your task is to replace element descriptions in the grounded action with element IDs from the list below based on the given task and screenshot.
The provided screenshot has labeled boxes with element IDs. Carefully observe the image and match the description with the corresponding labeled elements.
Think carefully about the context and available elements, find the best matching elements based on their descriptions and positions.
ONLY process these action types: click, type, scroll, drag_and_drop.

## Context Information

**Task Description**
{runner_result.get('next_action', 'Not provided')}

**Grounded Action**
{runner_result.get('grounded_action', 'Not provided')}

## Available UI Elements
{formatted_elements}

## Visual Analysis Instructions

1. **Understand the Task**: Task Description and Grounded Action described the task, understand what needs to be done
2. **Observe the Screenshot**: The image shows labeled boxes with element IDs overlaid on the UI elements
3. **Cross-Reference**: Use both the visual position in the image and the element list to find the correct match
4. **Element Matching Priority**: 
   - First try to match with INTERACTIVE ELEMENTS, if no suitable interactive element exists use NON-INTERACTIVE ELEMENTS
   - Match elements by:
     - Visual appearance in the labeled screenshot
     - Content text 
     - Element type 
     - Position and context

## Action Processing Rules

**CRITICAL**: Only replace the description/target in the Grounded Action with element ID (ONLY use a single numner for id). Do NOT modify any other parameters. 

- For agent.click(): Replace description with ID → agent.click(ID, num_clicks, button_type, hold_keys)
- For agent.type(): Replace description with ID → agent.type(ID, text, overwrite, enter)  
- For agent.scroll(): Replace description with ID → agent.scroll(ID, clicks, shift)
- For agent.drag_and_drop(): Replace descriptions with IDs → agent.drag_and_drop(start_ID, end_ID, hold_keys)

## Selection Strategy

- Look at the labeled boxes in the screenshot to identify elements visually
- Interactive elements are more reliable for automation
- Prefer edit fields over other elements for typing
- Prefer scrollbar for scroll
- Choose the most specific matching element based on visual confirmation

## Output Requirements

1. **No Element Found**: If no suitable element found, return "agent.fail()"
2. **Preserve Original Parameters**: Keep all original parameters except replace description with element ID, only use a single numner for id
3. **Single Line Output**: Return ONLY the modified grounded action as a single line
4. **Error Cases**: If grounded_action is agent.done() or agent.fail(), return unchanged
5. **positional parameters**you'd better use positional parameters instead of named parameters in the grounded_action code block

## Examples
Original: agent.click("Submit button", 1, 'left', [])
Modified: agent.click(5, 1, 'left', [])

Original: agent.type("search box", "Hello World", False, True)
Modified: agent.type(3, "Hello World", False, True)

Process the grounded action now by carefully examining the screenshot and element list
"""
        return prompt
    
    def _convert_ids_to_coordinates(self, llm_response: str, element_info_list: List[Dict]) -> str:
        """
        Convert element IDs in LLM response to actual coordinates
        
        Args:
            llm_response: LLM response with element IDs
            element_info_list: List of elements with coordinates
            
        Returns:
            Action string with coordinates
        """
        try:
            print(f"Converting IDs to coordinates in: {llm_response}")
            
            # Create ID to element mapping
            id_to_element = {elem.get('id'): elem for elem in element_info_list}
            
            # Parse the action to extract IDs
            if llm_response.startswith('agent.'):
                action_type = llm_response.split('(')[0].split('.')[1]
                
                if action_type in ['click', 'type', 'scroll']:
                    # Extract first parameter (should be ID)
                    params_str = llm_response.split('(')[1].split(')')[0]
                    params = [p.strip() for p in params_str.split(',')]
                    
                    if params and params[0].isdigit():
                        element_id = int(params[0])
                        
                        # Validate ID exists
                        if element_id in id_to_element:
                            element = id_to_element[element_id]
                            center_pixel = element.get('center_pixel', [0, 0])
                            x, y = int(center_pixel[0]), int(center_pixel[1])
                            
                            # Reconstruct action with coordinates
                            remaining_params = params[1:] if len(params) > 1 else []
                            new_params = [str(x), str(y)] + remaining_params
                            result = f"agent.{action_type}({', '.join(new_params)})"
                            
                            print(f"Successfully converted ID {element_id} to coordinates ({x}, {y})")
                            return result
                        else:
                            print(f"Element ID {element_id} not found in element list")
                            return "agent.fail()"
                
                elif action_type == 'drag_and_drop':
                    # Extract start and end IDs
                    params_str = llm_response.split('(')[1].split(')')[0]
                    params = [p.strip() for p in params_str.split(',')]
                    
                    if len(params) >= 2 and params[0].isdigit() and params[1].isdigit():
                        start_id, end_id = int(params[0]), int(params[1])
                        
                        # Validate both IDs exist
                        if start_id in id_to_element and end_id in id_to_element:
                            start_elem = id_to_element[start_id]
                            end_elem = id_to_element[end_id]
                            
                            start_center = start_elem.get('center_pixel', [0, 0])
                            end_center = end_elem.get('center_pixel', [0, 0])
                            
                            start_x, start_y = int(start_center[0]), int(start_center[1])
                            end_x, end_y = int(end_center[0]), int(end_center[1])
                            
                            # Reconstruct action
                            remaining_params = params[2:] if len(params) > 2 else ['[]']
                            new_params = [str(start_x), str(start_y), str(end_x), str(end_y)] + remaining_params
                            result = f"agent.drag_and_drop({', '.join(new_params)})"
                            
                            print(f"Successfully converted IDs {start_id}, {end_id} to coordinates")
                            return result
                        else:
                            print(f"One or both drag_and_drop IDs not found: {start_id}, {end_id}")
                            return "agent.fail()"
            
            print(f"Could not parse or convert LLM response: {llm_response}")
            return "agent.fail()"
            
        except Exception as e:
            print(f"Error converting IDs to coordinates: {e}")
            return "agent.fail()"
    
    def _try_accessibility_grounding(self, runner_result: Dict, screenshot_path: Optional[str] = None) -> str:
        """
        Try accessibility tree grounding as primary method
        
        Args:
            runner_result: Runner output containing action details
            screenshot_path: Optional screenshot path
            
        Returns:
            Grounded action with coordinates or agent.fail() if failed
        """
        try:
            # Get accessibility tree
            window_handle, window_title = self.tree_extractor.get_foreground_window_info()
            tree_data = None
            formatted_tree = "No accessibility tree available"
            print(f"[DEBUG] Formatted accessibility tree:\n{formatted_tree}")
            
            if not window_handle:
                print("No window handle available")
                return "agent.fail()"
            
            print(f"Attempting to get accessibility tree for window: {window_title}")
            
            # Try multiple times to get accessibility tree
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    tree_data = self.tree_extractor.get_window_accessibility_tree(window_handle)
                    if tree_data:
                        formatted_tree = self.tree_formatter.format_clickable_elements_for_llm(tree_data, window_title)
                        print(f"[DEBUG] Formatted accessibility tree:\n{formatted_tree}")
                        print(f"Accessibility tree found on attempt {attempt + 1}")
                        break
                    else:
                        print(f"Attempt {attempt + 1}: No accessibility tree data returned")
                        if attempt < max_retries - 1:
                            print("Retrying accessibility tree extraction...")
                            time.sleep(0.2)
                except Exception as e:
                    print(f"Attempt {attempt + 1}: Error getting accessibility tree: {e}")
                    if attempt < max_retries - 1:
                        print("Retrying accessibility tree extraction...")
                        time.sleep(0.2)
            
            if not tree_data:
                print("All attempts to get accessibility tree failed")
                return "agent.fail()"
            
            # Build grounding prompt
            grounding_prompt = self._build_accessibility_grounding_prompt(
                runner_result, formatted_tree, window_title or "Unknown Window"
            )
            
            # Call LLM to get grounded action
            grounded_action = self._generate_grounded_action(grounding_prompt, screenshot_path)
            print(f"[DEBUG] Accessibility LLM raw response: {grounded_action}")
            
            # Check if LLM failed to generate action
            if grounded_action == "agent.fail()":
                print("LLM failed to generate grounded action for accessibility tree")
                return "agent.fail()"
            
            # Validate coordinates in accessibility tree
            from utils.accessibility_tree import AccessibilityGrounder
            accessibility_grounder = AccessibilityGrounder()
            validated_action = accessibility_grounder.validate_accessibility_coordinates(grounded_action, tree_data)
            
            # Check if validation failed - this should also fallback to visual grounding
            if validated_action == "agent.fail()":
                print("Accessibility coordinate validation failed - will fallback to visual grounding")
                return "agent.fail()"
            
            print("Accessibility tree grounding validation successful")
            return validated_action
            
        except Exception as e:
            print(f"Error in accessibility tree grounding: {e}")
            return "agent.fail()"
    
    def _build_accessibility_grounding_prompt(self, runner_result: Dict, formatted_tree: str, window_title: str) -> str:
        """Build grounding prompt for accessibility tree grounding"""
        prompt = f"""
You are a very clever UI operator and able to use a mouse and keyboard to interact with the computer based on the given task and screenshot.

Your task is replace element descriptions in the grounded action with exact center coordinates from the clickable elements list based on the given task and screenshot.
Think carefully about the context and available elements, find the best matching elements based on their descriptions and positions.
Make sure to use the exact center coordinates provided in the clickable elements list, do not create new coordinates or modify them.
ONLY process these action types: click, type, scroll, drag_and_drop.

## Context Information

**Window Title:** {window_title}

**Task Description**
{runner_result.get('next_action', 'Not provided')}

**Grounded Action from Runner:**
{runner_result.get('grounded_action', 'Not provided')}

## Available Clickable Elements
{formatted_tree}



## Instructions

1. **Understand the Task**: Task Description and Grounded Action described the task, understand what needs to be done
2. **Observe the Screenshot**: The image shows labeled boxes with element IDs overlaid on the UI elements
3. **Cross-Reference**: Use both the visual position in the image and the element list to find the correct match
4. **Element Matching Priority**: 
   - First try to match with INTERACTIVE ELEMENTS, if no suitable interactive element exists use NON-INTERACTIVE ELEMENTS
   - Match elements by:
     - Visual appearance
     - Name/text content 
     - Element type 
     - AutomationID if available

**Coordinate System**: 
All coordinates are CENTER POINTS ready to use directly.
If you need position heuristics:
The coordinate format is (x, y) where:
- x is horizontal position: larger x values are further to the RIGHT
- y is vertical position: larger y values are further DOWN
- (0, 0) is at the top-left corner of the screen



**Action Processing**:
   - For agent.click(): Use center coordinates as agent.click(x, y, num_clicks, button_type, hold_keys)
   - For agent.type(): Use center coordinates as agent.type(x, y, text, overwrite, enter)
   - For agent.scroll(): Use center coordinates as agent.scroll(x, y, clicks, shift)
   - For agent.drag_and_drop(): Use two center coordinates as agent.drag_and_drop(start_x, start_y, end_x, end_y, hold_keys)

**No Element Found Cases**:
   - If no suitable element found, return "agent.fail()"

**Output Format**: Return ONLY the modified grounded action as a single line.
   you'd better use positional parameters instead of named parameters in the grounded_action code block
   Examples:
   - agent.click(150, 200, 1, 'left', [])
   - agent.type(300, 150, 'Hello World', False, True)
   - agent.scroll(400, 300, 3, False)
   - agent.drag_and_drop(100, 150, 200, 250, [])

**Error Cases**:
   - If grounded_action is agent.done() or agent.fail(): return unchanged
   - If action type not supported: return "agent.fail()"

Process the grounded action now:
"""
        return prompt
    

def take_action(grounded_action_code: str) -> bool:
    """
    Execute the grounded action code using pyautogui
    
    Args:
        grounded_action_code: Single line of Python-like agent code (e.g., "agent.click(150, 200, 1, 'left', [])")
        
    Returns:
        True if action executed successfully, False otherwise
    """
    try:
        # Create agent instance
        agent = Agent()
        
        # Clean the code
        code = grounded_action_code.strip()
        
        # Replace 'agent.' with 'agent.' to make it executable
        if code.startswith('agent.'):
            code = code[6:]  # Remove 'agent.' prefix
            
            # Parse and execute the action
            if code.startswith('click('):
                # Parse click parameters
                params = code[6:-1]  # Remove 'click(' and ')'
                args = safe_parse_params(params)  # Parse parameters safely using ast
                
                if len(args) >= 2:
                    # Check if first parameter is coordinate (int) or button text (str)
                    if isinstance(args[0], (int, float)):
                        # Coordinate-based click
                        x, y = int(args[0]), int(args[1])
                        num_clicks = int(args[2]) if len(args) > 2 else 1
                        button_type = args[3] if len(args) > 3 else 'left'
                        hold_keys = args[4] if len(args) > 4 else []
                        
                        return agent.click(x, y, num_clicks, button_type, hold_keys)
                    else:
                        # Button text-based click - not supported, should use coordinates
                        print(f"Text-based click not supported: {args[0]}")
                        return False
                    
            elif code.startswith('type('):
                # Parse type parameters
                params = code[5:-1]  # Remove 'type(' and ')'
                args = safe_parse_params(params)  # Parse parameters safely using ast
                
                if len(args) >= 3:
                    x, y = int(args[0]), int(args[1])
                    text = str(args[2])
                    overwrite = bool(args[3]) if len(args) > 3 else False
                    enter = bool(args[4]) if len(args) > 4 else False
                    
                    return agent.type(x, y, text, overwrite, enter)
                    
            elif code.startswith('scroll('):
                # Parse scroll parameters
                params = code[7:-1]  # Remove 'scroll(' and ')'
                args = safe_parse_params(params)  # Parse parameters safely using ast
                
                if len(args) >= 3:
                    x, y = int(args[0]), int(args[1])
                    clicks = int(args[2])
                    shift = bool(args[3]) if len(args) > 3 else False                    
                    return agent.scroll(x, y, clicks, shift)
                    
            elif code.startswith('drag_and_drop('):
                # Parse drag_and_drop parameters
                params = code[14:-1]  # Remove 'drag_and_drop(' and ')'
                args = safe_parse_params(params)  # Parse parameters safely using ast
                
                if len(args) >= 4:
                    start_x, start_y = int(args[0]), int(args[1])
                    end_x, end_y = int(args[2]), int(args[3])
                    hold_keys = args[4] if len(args) > 4 else []
                    
                    return agent.drag_and_drop(start_x, start_y, end_x, end_y, hold_keys)
            
            elif code.startswith('switch_applications('):
                # Parse switch_applications parameters
                params = code[20:-1]  # Remove 'switch_applications(' and ')'
                args = safe_parse_params(params)  # Parse parameters safely using ast
                
                if len(args) >= 1:
                    app_code = str(args[0])
                    return agent.switch_applications(app_code)
            
            elif code.startswith('open('):
                # Parse open parameters
                params = code[5:-1]  # Remove 'open(' and ')'
                args = safe_parse_params(params)  # Parse parameters safely using ast
                
                if len(args) >= 1:
                    app_or_filename = str(args[0])
                    return agent.open(app_or_filename)
            
            elif code.startswith('hotkey('):
                # Parse hotkey parameters
                params = code[7:-1]  # Remove 'hotkey(' and ')'
                args = safe_parse_params(params)  # Parse parameters safely using ast
                
                if len(args) >= 1:
                    keys = args[0] if isinstance(args[0], list) else args
                    return agent.hotkey(keys)
            
            elif code.startswith('hold_and_press('):
                # Parse hold_and_press parameters
                params = code[15:-1]  # Remove 'hold_and_press(' and ')'
                args = safe_parse_params(params)  # Parse parameters safely using ast
                
                if len(args) >= 2:
                    hold_keys = args[0]
                    press_keys = args[1]
                    return agent.hold_and_press(hold_keys, press_keys)
            
            elif code.startswith('wait('):
                # Parse wait parameters
                params = code[5:-1]  # Remove 'wait(' and ')'
                args = safe_parse_params(params)  # Parse parameters safely using ast
                
                if len(args) >= 1:
                    wait_time = float(args[0])
                    return agent.wait(wait_time)
            
            elif code.startswith('set_cell_values('):
                # Parse set_cell_values parameters
                params = code[16:-1]  # Remove 'set_cell_values(' and ')'
                args = safe_parse_params(params)  # Parse parameters safely using ast
                
                if len(args) >= 2:
                    cell_values = args[0]
                    app_name = str(args[1])
                    sheet_name = str(args[2]) if len(args) > 2 else None
                    return agent.set_cell_values(cell_values, app_name, sheet_name)
            
            elif code.startswith('highlight_text_span('):
                # Parse highlight_text_span parameters
                params = code[20:-1]  # Remove 'highlight_text_span(' and ')'
                args = safe_parse_params(params)  # Parse parameters safely using ast
                
                if len(args) >= 1:
                    # Check if we got a dictionary (named parameters) or list (positional parameters)
                    if len(args) == 1 and isinstance(args[0], dict):
                        # Named parameters: extract starting_phrase and ending_phrase from dict
                        param_dict = args[0]
                        starting_phrase = param_dict.get('starting_phrase', '')
                        ending_phrase = param_dict.get('ending_phrase', '')
                        
                        if starting_phrase and ending_phrase:
                            print(f"Parsed named parameters: starting_phrase='{starting_phrase}', ending_phrase='{ending_phrase}'")
                            return agent.highlight_text_span(starting_phrase, ending_phrase)
                        else:
                            print(f"Missing required parameters in: {param_dict}")
                            return False
                    elif len(args) >= 2:
                        # Positional parameters
                        starting_phrase = str(args[0])
                        ending_phrase = str(args[1])
                        print(f"Parsed positional parameters: starting_phrase='{starting_phrase}', ending_phrase='{ending_phrase}'")
                        return agent.highlight_text_span(starting_phrase, ending_phrase)
                    else:
                        print(f"Insufficient parameters for highlight_text_span: {args}")
                        return False
                else:
                    print(f"No parameters found for highlight_text_span")
                    return False
                    
            elif code.startswith('done('):
                return agent.done()
                
            elif code.startswith('fail('):
                return agent.fail()
        
        print(f"Unknown or invalid action code: {grounded_action_code}")
        return False
        
    except Exception as e:
        print(f"Error executing action '{grounded_action_code}': {e}")
        return False


def execute_grounded_runner_action(runner_result: Dict, screenshot_path: Optional[str] = None) -> bool:
    """
    Complete pipeline: Ground runner action and execute it
    
    Args:
        runner_result: Runner output containing grounded_action
        screenshot_path: Optional screenshot path
        
    Returns:
        True if action executed successfully, False otherwise
    """
    try:
        # Step 1: Ground the action (convert descriptions to coordinates)
        # Use global instance to avoid recreating and reloading models
        global _visual_grounder_instance
        if '_visual_grounder_instance' not in globals() or _visual_grounder_instance is None:
            _visual_grounder_instance = VisualGrounding()
        
        grounded_action = _visual_grounder_instance.ground_action(runner_result, screenshot_path)
        
        print(f"Grounded action: {grounded_action}")
        
        # Step 2: Execute the grounded action
        execution_success = take_action(grounded_action)
        
        return execution_success
        
    except Exception as e:
        print(f"Error in execute_grounded_runner_action: {e}")
        return False
