
import os
from io import BytesIO
from PIL import Image
from typing import List, Dict, Optional, Any, Tuple
from .my_grounding_agent_source import planner, runner, grounding
from .my_grounding_agent_source.main import AgentState
from .my_grounding_agent_source.logger_config import setup_logger


class OSWorldGroundingAgent:
    def __init__(self, vm_ip: str, server_port: int):
        self.vm_ip = vm_ip
        self.server_port = server_port
        self.state = AgentState()
        self.logger = setup_logger("osworld_adapter")
        self.logger.info("OSWorldGroundingAgent initialized.")
        

    def predict(self, instruction: str, obs: Dict, domain: str) -> Tuple[None, List[str]]:
        """
        Core method called by OS-World environment.
        """
        import time
        from io import BytesIO
        from PIL import Image

    
        if not hasattr(self.state, 'task_start_time'):
            self.state.task_start_time = time.time()
            
        
        if time.time() - self.state.task_start_time > 180:
            self.logger.warning("任务执行超时，强制结束")
            return None, ["FAIL"]

        
        if not self.state.task_description:
            self.state.task_description = instruction
            self.logger.info(f"New task started: {instruction}")

        
        screenshot_path = "temp_screenshot.png"
        try:
            image = Image.open(BytesIO(obs["screenshot"]))
            image.save(screenshot_path)
        except Exception as e:
            self.logger.error(f"Failed to save screenshot: {e}")
            return None, ["FAIL"]

        
        current_state_hash = self._get_screen_state_hash(obs)

        if not hasattr(self.state, 'repeated_state_count'):
            self.state.repeated_state_count = 0
        if not hasattr(self.state, 'fallback_methods'):
            self.state.fallback_methods = []

        if hasattr(self, 'last_state_hash') and self.last_state_hash == current_state_hash:
            self.state.repeated_state_count += 1
            self.logger.warning(f"状态重复次数: {self.state.repeated_state_count}")

            if self.state.repeated_state_count >= 3:
                instruction_lower = instruction.lower()

                
                if len(self.state.fallback_methods) >= 2 and self.state.repeated_state_count > 8:
                    self.logger.warning("所有备选方案均已尝试，任务可能无法完成")
                    return None, ["FAIL"]
        else:
            self.state.repeated_state_count = 0
            self.state.fallback_methods = []

        self.last_state_hash = current_state_hash

        # === PLANNER PHASE ===
        if not self.state.plan:
            self.logger.info("Running Planner Phase...")
            plan = planner.generate_plan(
                user_command=self.state.task_description,
                screenshot_path=screenshot_path,
                domain=domain,  
                future_steps=self.state.get_future_tasks(),
                completed_steps=self.state.completed_steps,
                failed_steps=self.state.failed_steps
            )

            if isinstance(plan, dict) and plan.get("task_completed"):
                return None, ["DONE"]
            elif isinstance(plan, list) and len(plan) > 0:
                self.state.plan = plan
                self.logger.info(f"New plan with {len(plan)} steps")
            else:
                self.logger.error("Planner failed")
                return None, ["FAIL"]

        # === RUNNER PHASE ===
        current_subtask = self.state.get_current_subtask()
        if not current_subtask:
            self.logger.warning("Plan finished but not DONE. Re-planning.")
            self.state.plan = []
            return self.predict(instruction, obs)

        self.logger.info(f"Running Runner for subtask: {current_subtask['description']}")
        action_result = runner.execute_runner_step(
            task_description=self.state.task_description,
            subtask=current_subtask,
            screenshot_path=screenshot_path,
            action_history=self.state.action_history
        )

        if not action_result:
            self.logger.error("Runner failed")
            return None, ["FAIL"]

        # === GROUNDING PHASE ===
        self.logger.info("Running Grounding Phase...")
        grounder = grounding.AccessibilityGrounding(vm_ip=self.vm_ip, server_port=self.server_port) 
        try:
            final_code = grounder.ground_action(action_result, screenshot_path)
        except Exception as e:
            import traceback
            self.logger.error(f"ERROR in ground_action: {e}")
            self.logger.error(traceback.format_exc())
            final_code = "agent.fail()"

        # === HANDLE FINAL CODE ===
        if 'agent.done()' in final_code:
            self.state.complete_current_subtask()
            if not self.state.plan:
                return None, ["DONE"]
            return None, ["WAIT"]
        elif 'agent.fail()' in final_code:
            self.state.fail_current_subtask()
            return None, ["FAIL"]
        else:
            self.state.action_history.add_action({
                'grounded_action': action_result.get('grounded_action', ''),
                'success': True
            })
            self.state.current_subtask_actions += 1

        self.logger.info(f"cooling time")
        time.sleep(5)
        return None, [final_code]


    def _get_screen_state_hash(self, obs):
        import hashlib
        if 'screenshot' in obs and obs['screenshot']:
            return hashlib.md5(obs['screenshot'][:10000]).hexdigest()
        return "no_screenshot"
    
    def reset(self, _logger=None):
        """Reset agent state for new task."""
        self.state = AgentState()
        self.logger.info("Agent state reset for new task.")