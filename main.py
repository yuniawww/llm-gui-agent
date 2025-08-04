#!/usr/bin/env python3
"""
AI Agent - Complete automation with grounding phase
Supports both automatic and debug modes for task execution
"""

import sys
import time
import os
import ctypes
from utils.screenshot import take_screenshot
from planner import generate_plan
from runner import execute_runner_step
from grounding import execute_grounded_runner_action
from history import ActionHistory
from logger_config import setup_logger, log_state_change, log_task_progress

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Execution mode settings
DEBUG_MODE = True  # Debug mode: pause after each phase for user confirmation
AUTO_MODE = False  # Automatic mode: run continuously without user input

# RAG settings
RAG_ENABLED = False  # RAG feature disabled by default

# Protection constants for automatic execution
MAX_ACTIONS_PER_SUBTASK = 8  # Maximum actions per subtask before replanning
MAX_ITERATIONS = 50  # Maximum total iterations before stopping

# Windows API constants for window management
SW_MINIMIZE = 6
SW_RESTORE = 9

# Phase confirmation messages
PHASE_CONFIRMATIONS = {
    'planner': "Press Enter to continue to Runner phase (or 'q' to quit)...",
    'runner': "Press Enter to continue to Grounding phase (or 'q' to quit)...",
    'grounding': "Press Enter to continue (or 'q' to quit)..."
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def minimize_console_window():
    """Minimize the console window on Windows"""
    try:
        if os.name == 'nt':  # Windows
            hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            if hwnd:
                ctypes.windll.user32.ShowWindow(hwnd, SW_MINIMIZE)
                main_logger.info("Console window minimized")
    except Exception as e:
        main_logger.warning(f"Could not minimize console window: {e}")

def restore_console_window():
    """Restore the console window on Windows"""
    try:
        if os.name == 'nt':  # Windows
            hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            if hwnd:
                ctypes.windll.user32.ShowWindow(hwnd, SW_RESTORE)
                main_logger.info("Console window restored")
    except Exception as e:
        main_logger.warning(f"Could not restore console window: {e}")

def wait_for_user_confirmation(phase_name: str):
    """Wait for user confirmation in debug mode"""
    if not DEBUG_MODE:
        return True
    
    message = PHASE_CONFIRMATIONS.get(phase_name, "Press Enter to continue (or 'q' to quit)...")
    user_input = input(f"\n[DEBUG] {message} ").strip().lower()
    
    if user_input == 'q':
        return False
    return True

# =============================================================================
# GLOBAL LOGGER SETUP
# =============================================================================

main_logger = setup_logger("main")

# =============================================================================
# AGENT STATE MANAGEMENT
# =============================================================================

class AgentState:
    """Manages Agent state for execution"""
    
    def __init__(self):
        self.task_description = ""
        self.plan = []
        self.failed_steps = []
        self.completed_steps = []  # Track completed subtasks
        self.action_history = ActionHistory()
        self.in_runner_mode = False
        self.max_actions_per_subtask = MAX_ACTIONS_PER_SUBTASK
        self.current_subtask_actions = 0
        self.logger = setup_logger("agent_state")
        
        self.logger.info("Agent state initialized")
    
    def get_current_subtask(self):
        """Get current subtask - always returns the first task in plan"""
        if self.plan:
            return self.plan[0]
        return None
    
    def get_future_tasks(self):
        """Get future tasks - returns all tasks except the first one as Dict objects"""
        if len(self.plan) > 1:
            return self.plan[1:]  # Return full Dict objects, not just descriptions
        return []
    
    def get_future_task_descriptions(self):
        """Get future task descriptions - returns string list for display"""
        if len(self.plan) > 1:
            return [task['description'] for task in self.plan[1:]]
        return []
    
    def complete_current_subtask(self):
        """Complete current subtask - move first task to completed_steps and remove from plan"""
        if self.plan:
            completed_task_dict = self.plan[0]  # Store full Dict object
            completed_task_desc = completed_task_dict['description']
            
            # Add to completed steps
            self.completed_steps.append(completed_task_dict)
            
            # Remove from current plan
            self.plan.pop(0)
            self.current_subtask_actions = 0
            
            log_task_progress(self.logger, "SUBTASK", completed_task_desc, "COMPLETED")
            
            # Clear action history for next subtask
            action_count = self.action_history.get_action_count()
            self.action_history.clear()
            self.logger.info(f"Cleared {action_count} actions from history")
            
            return True
        return False
    
    def fail_current_subtask(self):
        """Fail current subtask - move first task to failed_steps"""
        if self.plan:
            failed_task_dict = self.plan[0]  # Store full Dict object
            failed_task_desc = failed_task_dict['description']
            self.failed_steps.append(failed_task_dict)  # Store Dict, not just string
            self.plan.pop(0)
            self.current_subtask_actions = 0
            
            log_task_progress(self.logger, "SUBTASK", failed_task_desc, "FAILED", 
                            f"Total failed subtasks: {len(self.failed_steps)}")
            
            # Clear action history after failure
            action_count = self.action_history.get_action_count()
            self.action_history.clear()
            self.logger.info(f"Cleared {action_count} actions from history after failure")
            
            return True
        return False
    
    def should_replan_subtask(self):
        """Check if current subtask should be replanned due to too many actions"""
        if self.current_subtask_actions >= self.max_actions_per_subtask:
            self.logger.warning(f"Too many actions for current subtask ({self.current_subtask_actions}), replanning")
            return True
        return False

# =============================================================================
# STATUS DISPLAY FUNCTIONS
# =============================================================================

def print_status(state: AgentState):
    """Print current execution status"""
    print("\n" + "="*60)
    print(f"Task: {state.task_description}")
    
    if state.completed_steps:
        completed_descriptions = [step['description'] for step in state.completed_steps]
        print(f"Completed ({len(state.completed_steps)}): " + ", ".join(completed_descriptions))
    
    if state.failed_steps:
        failed_descriptions = [step['description'] for step in state.failed_steps]
        print(f"Failed ({len(state.failed_steps)}): " + ", ".join(failed_descriptions))
    
    current_subtask = state.get_current_subtask()
    if current_subtask:
        print(f"Current: {current_subtask['description']} (Actions: {state.current_subtask_actions})")
    
    future_tasks = state.get_future_task_descriptions()
    if future_tasks:
        if len(future_tasks) > 2:
            print(f"Remaining ({len(future_tasks)}): " + ", ".join(future_tasks[:2]) + "...")
        else:
            print(f"Remaining ({len(future_tasks)}): " + ", ".join(future_tasks))
    
    print("="*60)

def print_phase_header(phase_name: str, description: str = ""):
    """Print formatted phase header"""
    print(f"\n[{phase_name.upper()} PHASE] {description}")

# =============================================================================
# PHASE EXECUTION FUNCTIONS
# =============================================================================

def run_planner_phase(state: AgentState):
    """Run planning phase"""
    main_logger.info("Starting planner phase")
    print_phase_header("PLANNER", "Analyzing current state and generating plan...")
    
    try:
        # Take screenshot
        screenshot_path = take_screenshot()
        main_logger.info(f"Screenshot saved: {screenshot_path}")
        print(f"Screenshot saved: {screenshot_path}")
        
        # Generate or update plan
        print("Generating/updating plan...")
        main_logger.info("Calling planner API")
        
        try:
            plan = generate_plan(
                user_command=state.task_description,
                screenshot_path=screenshot_path,
                future_steps=state.get_future_tasks(),
                completed_steps=state.completed_steps,
                failed_steps=state.failed_steps
            )
            main_logger.info("Planner API call succeeded")
        except Exception as e:
            main_logger.error(f"Planner API failed: {str(e)}")
            print(f"Planner API Error: {str(e)}")
            return False
        
        # Check if planner returned valid result
        if plan is None:
            main_logger.error("No valid plan generated (None returned)")
            print("No valid plan generated.")
            return False
        
        # Check if task is completed (planner returns completion marker)
        if isinstance(plan, dict) and plan.get("task_completed") is True:
            completion_message = plan.get("completion_message", "Task completed")
            main_logger.info(f"Task completed - planner returned completion marker: {completion_message}")
            print(f"🎉 Task completed! {completion_message}")
            return "completed"
        
        # Check if task is completed (planner returns empty list - backward compatibility)
        # But distinguish between "task completed" and "planner failed"
        if len(plan) == 0:
            # Check if this is likely a planner failure (no existing plan) vs task completion
            if not state.plan and state.current_subtask_actions == 0:
                # This looks like a planner failure on initial planning
                main_logger.error("Planner failed to generate initial plan - returned empty list")
                print("ERROR: Planner failed to generate plan. This may be due to:")
                print("  - API issues or rate limits")
                print("  - JSON parsing failures after multiple retries")
                print("  - Invalid prompt or response format")
                return False
            else:
                # This might be legitimate task completion (fallback for old format)
                main_logger.info("Task completed - planner returned empty plan")
                print("🎉 Task completed! Planner returned empty plan.")
                return "completed"
        
        # Ensure plan is a list for normal operation
        if not isinstance(plan, list):
            main_logger.error(f"Invalid plan format - expected list, got {type(plan)}")
            print("ERROR: Invalid plan format received from planner.")
            return False
        
        # Update state with new plan
        state.plan = plan
        log_task_progress(main_logger, "PLANNING", state.task_description, "COMPLETED", 
                         "Plan generated successfully")
        
        print(f"\nPlan updated with {len(plan)} subtasks:")
        for i, subtask in enumerate(plan, 1):
            description = subtask.get('description', f'Subtask {i}')
            print(f"  {i}. {description}")
        
        return True
        
    except Exception as e:
        main_logger.error(f"Unexpected error in planner phase: {str(e)}")
        print(f"Unexpected error in planner: {e}")
        return False

def run_runner_phase(state: AgentState):
    """Run execution phase"""
    current_subtask = state.get_current_subtask()
    if not current_subtask:
        main_logger.warning("No current subtask available in runner phase")
        print("No current subtask available.")
        return False
    
    subtask_desc = current_subtask['description']
    main_logger.info(f"Starting runner phase for subtask: {subtask_desc}")
    print_phase_header("RUNNER", f"Working on: {subtask_desc}")
    
    try:
        # Take screenshot
        screenshot_path = take_screenshot()
        main_logger.info(f"Screenshot taken: {screenshot_path}")
        print(f"Screenshot taken: {screenshot_path}")
        
        # Generate action using runner
        print("Analyzing screen and generating action...")
        main_logger.info("Calling runner API")
        
        try:
            action_result = execute_runner_step(
                task_description=state.task_description,
                subtask=current_subtask,
                screenshot_path=screenshot_path,
                future_tasks=state.get_future_task_descriptions(),  # Use string list for runner
                action_history=state.action_history
            )
            main_logger.info("Runner API call succeeded")
        except Exception as e:
            main_logger.error(f"Runner API failed: {str(e)}")
            print(f"Runner API Error: {str(e)}")
            return False
        
        if not action_result:
            main_logger.error("Failed to generate action")
            print("Failed to generate action.")
            if action_result is None:
                print("ERROR: Runner failed after multiple retries. This may be due to:")
                print("  - API issues or rate limits")
                print("  - JSON parsing failures")
                print("  - Invalid response format")
            return False
        
        # Validate action_result completeness
        required_fields = ["screen_analysis", "previous_action_verification", "next_action", "grounded_action"]
        missing_fields = [field for field in required_fields if field not in action_result]
        
        if missing_fields:
            main_logger.warning(f"Action result missing fields: {missing_fields}")
            for field in missing_fields:
                action_result[field] = "N/A (missing from response)"
        
        # Display action information
        print("\n[ACTION ANALYSIS]")
        print(f"Screen Analysis: {action_result.get('screen_analysis', 'N/A')}")
        print(f"Previous Action Status: {action_result.get('previous_action_verification', 'N/A')}")
        print(f"Next Action: {action_result.get('next_action', 'N/A')}")
        print(f"Grounded Action: {action_result.get('grounded_action', 'N/A')}")
        
        # Log action details
        main_logger.info(f"Generated action: {action_result.get('grounded_action', 'N/A')}")
        
        # Check if subtask is completed or failed
        grounded_action = action_result.get('grounded_action', '')
        if 'agent.done()' in grounded_action:
            print(f"\nSubtask completed: {subtask_desc}")
            log_task_progress(main_logger, "SUBTASK", subtask_desc, "COMPLETED")
            state.complete_current_subtask()
            state.in_runner_mode = False
            return "completed"
        elif 'agent.fail()' in grounded_action:
            print(f"\nSubtask failed: {subtask_desc}")
            print("Returning to planner for re-planning...")
            log_task_progress(main_logger, "SUBTASK", subtask_desc, "FAILED")
            state.fail_current_subtask()
            state.in_runner_mode = False
            return "failed"
        
        # Store action result for grounding phase
        state.current_action_result = action_result
        state.current_screenshot_path = screenshot_path
        
        return "ready_for_grounding"
        
    except Exception as e:
        main_logger.error(f"Unexpected error in runner phase: {str(e)}")
        print(f"Unexpected error in runner: {e}")
        return False

def run_grounding_phase(state: AgentState):
    """Run grounding phase - execute the grounded action"""
    main_logger.info("Starting grounding phase")
    print_phase_header("GROUNDING", "Executing action...")
    
    try:
        action_result = getattr(state, 'current_action_result', None)
        screenshot_path = getattr(state, 'current_screenshot_path', None)
        
        if not action_result:
            print("No action result available for grounding")
            return False
        
        # Minimize console window during grounding to avoid interference
        minimize_console_window()
        
        # Wait for target window to gain focus after console minimization
        print("Waiting for target window to gain focus...")
        time.sleep(1.0)  # Give target window time to become active
        
        # Execute the grounded action
        execution_success = execute_grounded_runner_action(action_result, screenshot_path)
        
        # Always restore console window after grounding
        restore_console_window()
        
        # Process execution result
        if execution_success:
            print("Action executed successfully")
            main_logger.info("Action executed successfully")
            state.current_subtask_actions += 1
            
            # Add successful action to history
            state.action_history.add_action({
                'grounded_action': action_result.get('grounded_action', ''),
                'screen_analysis': action_result.get('screen_analysis', ''),
                'previous_action_verification': action_result.get('previous_action_verification', ''),
                'next_action': action_result.get('next_action', ''),
                'success': True
            })
            
            return "executed"
        else:
            print("Action execution failed")
            main_logger.error("Action execution failed")
            state.current_subtask_actions += 1  # Count failed actions too
            
            # Add failed action to history
            state.action_history.add_action({
                'grounded_action': action_result.get('grounded_action', ''),
                'screen_analysis': action_result.get('screen_analysis', ''),
                'previous_action_verification': action_result.get('previous_action_verification', ''),
                'next_action': action_result.get('next_action', ''),
                'success': False
            })
            
            return "failed"
            
    except Exception as e:
        # Ensure console window is restored even on error
        restore_console_window()
        main_logger.error(f"Error during grounding execution: {str(e)}")
        print(f"Error during action execution: {e}")
        state.current_subtask_actions += 1  # Count exception failures too
        
        # Add exception failure to history
        state.action_history.add_action({
            'grounded_action': action_result.get('grounded_action', '') if action_result else 'Unknown',
            'screen_analysis': 'Exception during execution',
            'previous_action_verification': 'N/A',
            'next_action': 'N/A', 
            'success': False,
            'error': str(e)
        })
        
        return "failed"

# =============================================================================
# MAIN EXECUTION LOGIC
# =============================================================================

def run_agent_execution(task_description: str, max_iterations: int = MAX_ITERATIONS):
    """Run the agent with complete planning, running, and grounding phases"""
    main_logger.info(f"Starting agent execution for task: {task_description}")
    print(f"\n=== AGENT EXECUTION ===")
    print(f"Task: {task_description}")
    print(f"Mode: {'DEBUG' if DEBUG_MODE else 'AUTOMATIC'}")
    print(f"Max iterations: {max_iterations}")
    
    # Initialize agent state
    state = AgentState()
    state.task_description = task_description
    
    # Execution states
    current_phase = "planning"  # "planning", "running", or "grounding"
    iteration_count = 0
    
    log_task_progress(main_logger, "MAIN_TASK", task_description, "STARTED")
    
    while iteration_count < max_iterations:
        iteration_count += 1
        main_logger.info(f"=== ITERATION {iteration_count} ===")
        print(f"\n=== ITERATION {iteration_count} ===")
        print_status(state)
        
        try:
            if current_phase == "planning":
                result = run_planner_phase(state)
                
                if not wait_for_user_confirmation('planner'):
                    return False
                
                if result == "completed":
                    print(f"\n=== TASK COMPLETED ===")
                    print(f"Task: {task_description}")
                    print(f"Total iterations: {iteration_count}")
                    print(f"Completed subtasks: {len(state.completed_steps)}")
                    print(f"Failed subtasks: {len(state.failed_steps)}")
                    log_task_progress(main_logger, "MAIN_TASK", task_description, "COMPLETED")
                    return True
                elif result:
                    state.in_runner_mode = True
                    current_phase = "running"
                    print("Switching to runner phase...")
                else:
                    print("Planning failed, retrying...")
                    time.sleep(2)  # Brief pause before retry
                    
            elif current_phase == "running":
                # Check if current subtask should be replanned
                if state.should_replan_subtask():
                    print("Too many actions for current subtask, replanning...")
                    current_phase = "planning"
                    state.in_runner_mode = False
                    state.current_subtask_actions = 0
                    continue
                
                result = run_runner_phase(state)
                
                if not wait_for_user_confirmation('runner'):
                    return False
                
                if result == "completed":
                    print("Subtask completed, returning to planner for evaluation...")
                    # Always return to planner after subtask completion to reassess
                    current_phase = "planning"
                    state.in_runner_mode = False
                elif result == "failed":
                    print("Subtask failed, returning to planner...")
                    current_phase = "planning"
                    state.in_runner_mode = False
                elif result == "ready_for_grounding":
                    print("Action generated, moving to grounding...")
                    current_phase = "grounding"
                else:
                    print("Runner failed, retrying...")
                    time.sleep(2)  # Brief pause before retry
                    
            elif current_phase == "grounding":
                result = run_grounding_phase(state)
                
                if not wait_for_user_confirmation('grounding'):
                    return False
                
                if result == "executed":
                    print("Action executed successfully, continuing...")
                    current_phase = "running"  # Continue with runner
                    time.sleep(1)  # Brief pause between actions
                else:
                    print("Grounding failed")
                    
                    # Check if too many actions have been attempted for this subtask
                    if state.should_replan_subtask():
                        print("Too many failed actions, returning to planner for re-planning...")
                        current_phase = "planning"
                        state.in_runner_mode = False
                        state.current_subtask_actions = 0
                    else:
                        print("Continuing with runner to generate new action...")
                        current_phase = "running"  # Go back to runner for new action
                        time.sleep(1)  # Brief pause before retry
            
        except KeyboardInterrupt:
            print("\n\nExecution interrupted by user (Ctrl+C)")
            restore_console_window()
            return False
        except Exception as e:
            main_logger.error(f"Unexpected error in iteration {iteration_count}: {str(e)}")
            print(f"Unexpected error: {e}")
            restore_console_window()
            time.sleep(2)  # Brief pause before retry
    
    # Maximum iterations reached
    print(f"\n=== MAXIMUM ITERATIONS REACHED ===")
    print(f"Task: {task_description}")
    print(f"Total iterations: {iteration_count}")
    print(f"Completed subtasks: {len(state.completed_steps)}")
    print(f"Failed subtasks: {len(state.failed_steps)}")
    print("Task not completed within maximum iterations")
    
    main_logger.warning(f"Task not completed within {max_iterations} iterations")
    return False

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function for agent execution"""
    main_logger.info("=== AI Agent Started ===")
    print("=== AI Agent with Grounding ===")
    print("Complete automation with planning, running, and grounding phases")
    print(f"Mode: {'DEBUG' if DEBUG_MODE else 'AUTOMATIC'}")
    print(f"RAG: {'ENABLED' if RAG_ENABLED else 'DISABLED'}")
    print("\nCommands:")
    print("  - Enter task description to start execution")
    print("  - 'quit' to exit")
    if DEBUG_MODE:
        print("  - In debug mode: press Enter after each phase to continue")
    
    while True:
        try:
            user_input = input("\nEnter your task description (or 'quit' to exit): ").strip()
            
            if user_input.lower() == 'quit':
                main_logger.info("User requested quit")
                print("Goodbye!")
                break
            elif not user_input:
                print("Please enter a task description.")
                continue
            
            print(f"\nStarting execution for: {user_input}")
            print("=" * 60)
            
            # Run agent execution
            success = run_agent_execution(user_input)
            
            if success:
                print("\n" + "=" * 60)
                print("TASK COMPLETED SUCCESSFULLY!")
                print("=" * 60)
            else:
                print("\n" + "=" * 60)
                print("TASK FAILED OR INTERRUPTED")
                print("=" * 60)
            
            # Ask if user wants to continue
            continue_input = input("\nDo you want to start another task? (y/N): ").strip().lower()
            if continue_input != 'y':
                break
                
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user (Ctrl+C)")
            restore_console_window()
            user_input = input("Do you want to quit? (y/N): ").strip().lower()
            if user_input == 'y':
                main_logger.info("User confirmed quit after Ctrl+C")
                break
        except Exception as e:
            main_logger.error(f"Unexpected error in main loop: {str(e)}")
            print(f"An error occurred: {e}")
            print("Continuing...")
            restore_console_window()
    
    main_logger.info("=== AI Agent Terminated ===")

# =============================================================================
# DEBUG AND TESTING FUNCTIONS
# =============================================================================

def test_data_type_consistency():
    """Test function to verify data type consistency between main.py and planner.py"""
    print("=== Data Type Consistency Test ===")
    
    # Create a test state
    state = AgentState()
    state.plan = [
        {"subtask": 1, "description": "First task"},
        {"subtask": 2, "description": "Second task"},
        {"subtask": 3, "description": "Third task"}
    ]
    state.failed_steps = [
        {"subtask": 0, "description": "Previously failed task"}
    ]
    
    print("Test plan:")
    for task in state.plan:
        print(f"  {task}")
    
    print(f"\nFailed steps: {state.failed_steps}")
    
    # Test the methods
    print("\n--- Method outputs ---")
    future_tasks = state.get_future_tasks()
    future_descriptions = state.get_future_task_descriptions()
    
    print(f"get_future_tasks() returns: {future_tasks}")
    print(f"Type: {type(future_tasks)}")
    if future_tasks:
        print(f"First element type: {type(future_tasks[0])}")
    
    print(f"\nget_future_task_descriptions() returns: {future_descriptions}")
    print(f"Type: {type(future_descriptions)}")
    if future_descriptions:
        print(f"First element type: {type(future_descriptions[0])}")
    
    print(f"\nfailed_steps contains: {state.failed_steps}")
    print(f"Type: {type(state.failed_steps)}")
    if state.failed_steps:
        print(f"First element type: {type(state.failed_steps[0])}")
    
    # Test what would be passed to planner
    print("\n--- What gets passed to planner ---")
    print("future_steps parameter:")
    import json
    try:
        serialized_future = json.dumps(future_tasks)
        print(f"  JSON: {serialized_future}")
        print("  ✅ JSON serialization successful")
    except Exception as e:
        print(f"  ❌ JSON serialization failed: {e}")
    
    print("failed_steps parameter:")
    try:
        serialized_failed = json.dumps(state.failed_steps)
        print(f"  JSON: {serialized_failed}")
        print("  ✅ JSON serialization successful")
    except Exception as e:
        print(f"  ❌ JSON serialization failed: {e}")
    
    print("\n--- Simulated planner prompt context ---")
    print("**Remaining Subtasks**:", json.dumps(future_tasks))
    print("**Failed Subtasks**:", json.dumps(state.failed_steps))
    
    print("=== Test Complete ===\n")

if __name__ == "__main__":
    main()
