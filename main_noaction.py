# main.py

import sys
from utils.screenshot import take_screenshot
from planner import generate_plan
from runner import execute_runner_step
from history import ActionHistory
from logger_config import setup_logger, log_state_change, log_task_progress

# 设置全局日志器
main_logger = setup_logger("main")

class AgentState:
    """管理Agent的状态"""
    
    def __init__(self):
        self.task_description = ""
        self.plan = []
        self.completed_steps = []
        self.failed_steps = []  # 新增：记录失败的子任务
        self.action_history = ActionHistory()
        self.in_runner_mode = False
        self.logger = setup_logger("agent_state")
        
        self.logger.info("Agent state initialized")
    
    def get_current_subtask(self):
        """获取当前子任务 - 总是返回计划中的第一个任务"""
        if self.plan:
            return self.plan[0]
        return None
    
    def get_future_tasks(self):
        """获取未来的任务 - 返回除第一个之外的所有任务"""
        if len(self.plan) > 1:
            return [task['description'] for task in self.plan[1:]]
        return []
    
    def complete_current_subtask(self):
        """完成当前子任务 - 将第一个任务移到completed_steps"""
        if self.plan:
            completed_task = self.plan[0]['description']
            self.completed_steps.append(completed_task)
            self.plan.pop(0)  # 移除已完成的第一个任务
            
            # 记录日志
            log_task_progress(self.logger, "SUBTASK", completed_task, "COMPLETED")
            
            # 清空历史记录（考虑是否需要保留一些历史）
            action_count = self.action_history.get_action_count()
            self.action_history.clear()
            self.logger.info(f"Cleared {action_count} actions from history")
            
            return True
        return False
    
    def fail_current_subtask(self):
        """失败当前子任务 - 将第一个任务移到failed_steps"""
        if self.plan:
            failed_task = self.plan[0]['description']
            self.failed_steps.append(failed_task)
            self.plan.pop(0)  # 移除失败的第一个任务
            
            # 记录日志
            log_task_progress(self.logger, "SUBTASK", failed_task, "FAILED", 
                            f"Total failed subtasks: {len(self.failed_steps)}")
            
            # 清空历史记录
            action_count = self.action_history.get_action_count()
            self.action_history.clear()
            self.logger.info(f"Cleared {action_count} actions from history after failure")
            
            return True
        return False


def print_status(state: AgentState):
    """打印当前状态"""
    print("\n" + "="*60)
    print(f" Task: {state.task_description}")
    
    if state.completed_steps:
        print(" Completed: " + ", ".join(state.completed_steps))
    
    if state.failed_steps:
        print(" Failed: " + ", ".join(state.failed_steps))
    
    current_subtask = state.get_current_subtask()
    if current_subtask:
        print(f" Current: {current_subtask['description']}")
    
    future_tasks = state.get_future_tasks()
    if future_tasks:
        if len(future_tasks) > 2:
            print(" Upcoming: " + ", ".join(future_tasks[:2]) + "...")
        else:
            print(" Upcoming: " + ", ".join(future_tasks))
    
    print("="*60)


def run_planner_phase(state: AgentState):
    """运行规划阶段"""
    main_logger.info("Starting planner phase")
    print("\n [PLANNER MODE]")
    print("Taking screenshot and analyzing current state...")
    
    try:
        # 截取屏幕
        screenshot_path = take_screenshot()
        main_logger.info(f"Screenshot saved: {screenshot_path}")
        print(" Screenshot saved: " + screenshot_path)
        
        # 生成或更新计划
        print("\n Generating/updating plan...")
        main_logger.info("Calling planner API")
        
        # 使用API处理器调用planner
        def planner_call():
            return generate_plan(
                user_command=state.task_description,
                screenshot_path=screenshot_path,
                completed_steps=state.completed_steps,
                future_steps=state.get_future_tasks(),
                failed_steps=state.failed_steps
            )
        
        # 直接调用planner，简单错误处理
        try:
            plan = planner_call()
            main_logger.info("Planner API call succeeded")
        except Exception as e:
            # API失败，简单处理
            main_logger.error(f"Planner API failed: {str(e)}")
            print(f" API Error: {str(e)}")
            print(" Planner failed, please try again.")
            return False
        
        # 检查planner是否成功返回（包括空列表）
        if plan is None:
            main_logger.error("No valid plan generated (None returned)")
            print(" No valid plan generated.")
            return False
        
        # 检查是否任务完成（planner返回空列表）
        if len(plan) == 0:
            main_logger.info("Task completed - planner returned empty plan")
            print(" Task completed! Planner returned empty plan.")
            return "completed"
          # 更新状态
        state.plan = plan
        log_task_progress(main_logger, "PLANNING", state.task_description, "COMPLETED", 
                         "Plan generated successfully")
        
        print("\n Plan updated with subtasks:")
        for i, subtask in enumerate(plan, 1):
            description = subtask.get('description', f'Subtask {i}')
            print(f"  ⏳ {i}. {description}")
        
        return True
        
    except Exception as e:
        main_logger.error(f"Unexpected error in planner phase: {str(e)}")
        print(f" Unexpected error: {e}")
        print(" Planner failed completely.")
        return False


def run_runner_phase(state: AgentState):
    """运行执行阶段"""
    current_subtask = state.get_current_subtask()
    if not current_subtask:
        main_logger.warning("No current subtask available in runner phase")
        print(" No current subtask available.")
        return False
    
    subtask_desc = current_subtask['description']
    main_logger.info(f"Starting runner phase for subtask: {subtask_desc}")
    print(" RUNNER MODE - Working on: " + subtask_desc)
    
    try:
        # 截取屏幕
        screenshot_path = take_screenshot()
        main_logger.info(f"Screenshot taken: {screenshot_path}")
        print(" Screenshot taken: " + screenshot_path)
        
        # 生成动作
        print("\n Analyzing screen and generating action...")
        main_logger.info("Calling runner API")
        
        # 使用API处理器调用runner
        def runner_call():
            return execute_runner_step(
                task_description=state.task_description,
                subtask=current_subtask,
                screenshot_path=screenshot_path,
                future_tasks=state.get_future_tasks(),
                action_history=state.action_history
            )
        
        # 直接调用runner，简单错误处理
        try:
            action_result = runner_call()
            main_logger.info("Runner API call succeeded")
        except Exception as e:
            # API失败，简单处理
            main_logger.error(f"Runner API failed: {str(e)}")
            print(f" API Error: {str(e)}")
            print(" Runner failed.")
            return False
        
        if not action_result:
            main_logger.error("Failed to generate action")
            print(" Failed to generate action.")
            return False
        
        # 验证action_result的完整性
        required_fields = ["screen_analysis", "previous_action_verification", "next_action", "grounded_action"]
        missing_fields = [field for field in required_fields if field not in action_result]
        
        if missing_fields:
            main_logger.warning(f"Action result missing fields: {missing_fields}")
            for field in missing_fields:
                action_result[field] = "N/A (missing from response)"
        # 显示动作信息
        print("\n [ACTION ANALYSIS]")
        print(" Screen Analysis: " + action_result.get('screen_analysis', 'N/A'))
        print(" Previous Action Status: " + action_result.get('previous_action_verification', 'N/A'))
        print(" Next Action: " + action_result.get('next_action', 'N/A'))
        print(" Grounded Action: " + action_result.get('grounded_action', 'N/A'))
        
        # 记录动作详情到日志
        main_logger.info(f"Generated action: {action_result.get('grounded_action', 'N/A')}")
        
        # 检查是否任务完成
        grounded_action = action_result.get('grounded_action', '')
        if 'agent.done()' in grounded_action:
            print("\n Subtask completed: " + subtask_desc)
            log_task_progress(main_logger, "SUBTASK", subtask_desc, "COMPLETED")
            state.complete_current_subtask()
            state.in_runner_mode = False
            return True
        elif 'agent.fail()' in grounded_action:
            print("\n Subtask failed: " + subtask_desc)
            print(" Returning to planner for re-planning...")
            log_task_progress(main_logger, "SUBTASK", subtask_desc, "FAILED")
            
            # 记录失败的子任务
            state.fail_current_subtask()
            state.in_runner_mode = False
            return True
        
        main_logger.info("Action generated successfully, continuing execution")
        return True
        
    except Exception as e:
        main_logger.error(f"Unexpected error in runner phase: {str(e)}")
        print(f" Unexpected error: {e}")
        print(" Runner failed completely.")
        return False


# 状态枚举
class AppState:
    IDLE = "idle"
    PLANNING = "planning"
    RUNNING = "running"
    COMPLETED = "completed"

def handle_idle_state(state: AgentState):
    """处理空闲状态 - 等待任务输入"""
    user_input = input("\n Enter your task description: ").strip()
    
    if user_input.lower() == 'quit':
        main_logger.info("User requested quit from idle state")
        return 'quit', None
    elif not user_input:
        print(" Please enter a task description.")
        return AppState.IDLE, None
    
    state.task_description = user_input
    log_task_progress(main_logger, "MAIN_TASK", user_input, "STARTED")
    log_state_change(main_logger, AppState.IDLE, AppState.PLANNING, "User provided task description")
    return AppState.PLANNING, None

def handle_planning_state(state: AgentState):
    """处理规划状态"""
    result = run_planner_phase(state)
    if result == "completed":
        # Planner返回空列表，任务已完成
        log_state_change(main_logger, AppState.PLANNING, AppState.COMPLETED, "Task completed by planner")
        return AppState.COMPLETED, None
    elif result:
        state.in_runner_mode = True
        print_status(state)
        print("\n Type 'f' to start executing the first subtask")
        log_state_change(main_logger, AppState.PLANNING, AppState.RUNNING, "Plan generated successfully")
        return AppState.RUNNING, None
    else:
        print(" Planning failed. Please try again.")
        log_task_progress(main_logger, "PLANNING", state.task_description, "FAILED")
        state.task_description = ""
        log_state_change(main_logger, AppState.PLANNING, AppState.IDLE, "Planning failed")
        return AppState.IDLE, None

def handle_running_state(state: AgentState):
    """处理运行状态"""
    print_status(state)
    
    user_input = input("\n Enter command (f/status/reset/quit): ").strip().lower()
    
    if user_input == 'quit':
        main_logger.info("User requested quit from running state")
        return 'quit', None
    elif user_input == 'reset':
        main_logger.info("User requested reset")
        log_state_change(main_logger, AppState.RUNNING, AppState.IDLE, "User reset")
        return AppState.IDLE, AgentState()
    elif user_input == 'status':
        # 显示详细状态
        print("\n [Status Information]")
        print(f" Task: {state.task_description}")
        if state.plan:
            print(" Current subtask: Working on current task")
        print(" Completed steps: " + (", ".join(state.completed_steps) if state.completed_steps else "None"))
        print(" Failed steps: " + (", ".join(state.failed_steps) if state.failed_steps else "None"))
        return AppState.RUNNING, None
    elif user_input == 'f':
        return handle_f_command(state)
    else:
        print(" Unknown command. Use 'f', 'status', 'reset', or 'quit'.")
        return AppState.RUNNING, None

def handle_f_command(state: AgentState):
    """处理'f'命令执行"""
    if state.in_runner_mode:
        # 运行Runner阶段
        if run_runner_phase(state):
            if not state.in_runner_mode:
                # 子任务完成或失败，回到planner重新评估
                return AppState.PLANNING, None
        else:
            print(" Runner execution failed.")
        return AppState.RUNNING, None
    else:
        # 运行Planner阶段
        result = run_planner_phase(state)
        if result == "completed":
            return AppState.COMPLETED, None
        elif result:
            state.in_runner_mode = True
            return AppState.RUNNING, None
        else:
            print(" Planning failed.")
            return AppState.RUNNING, None

def handle_completed_state(state: AgentState):
    """处理完成状态"""
    user_input = input("\n Task completed! Enter new task or 'quit': ").strip()
    
    if user_input.lower() == 'quit':
        return 'quit', None
    elif user_input:
        new_state = AgentState()
        new_state.task_description = user_input
        return AppState.PLANNING, new_state
    else:
        return AppState.IDLE, AgentState()

def main():
    main_logger.info("=== AI Agent Started ===")
    print("=== AI Agent Interactive CLI ===")
    print("Commands:")
    print("  - Enter task description to start")
    print("  - 'f' to continue with next action")
    print("  - 'quit' to exit")
    print("  - 'status' to see current state")
    print("  - 'reset' to start over")
    
    current_app_state = AppState.IDLE
    agent_state = AgentState()
    
    # 状态处理器映射
    state_handlers = {
        AppState.IDLE: handle_idle_state,
        AppState.PLANNING: handle_planning_state,
        AppState.RUNNING: handle_running_state,
        AppState.COMPLETED: handle_completed_state
    }
    
    main_logger.info(f"Starting main loop in state: {current_app_state}")
    
    while True:
        try:
            # 根据当前状态调用对应的处理器
            handler = state_handlers.get(current_app_state)
            if not handler:
                main_logger.error(f"Unknown state: {current_app_state}")
                print(f"Unknown state: {current_app_state}")
                break
            
            # 执行状态处理器
            next_state, new_agent_state = handler(agent_state)
            
            # 更新状态
            if next_state == 'quit':
                main_logger.info("Application terminated normally")
                print(" Goodbye!")
                break
            
            # 记录状态变化
            if next_state != current_app_state:
                log_state_change(main_logger, current_app_state, next_state)
            
            current_app_state = next_state
            if new_agent_state is not None:
                agent_state = new_agent_state
        
        except KeyboardInterrupt:
            main_logger.warning("Operation cancelled by user (Ctrl+C)")
            print("\n\n Operation cancelled by user.")
            user_input = input("Do you want to quit? (y/N): ").strip().lower()
            if user_input == 'y':
                main_logger.info("User confirmed quit after Ctrl+C")
                break
        except Exception as e:
            main_logger.error(f"Unexpected error in main loop: {str(e)}")
            print(f" An error occurred: {e}")
            print(" Continuing...")


if __name__ == "__main__":
    main()
