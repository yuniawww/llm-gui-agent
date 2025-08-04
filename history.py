# history.py
from typing import List, Dict, Optional, Any
from datetime import datetime

class ActionHistory:
    """manage action history with timestamps and formatted output"""
    
    def __init__(self, max_history: int = 10):
        self.actions = []
        self.max_history = max_history
    
    def add_action(self, action_data: Dict[str, Any]):
        """add an action to the history, automatically adding a timestamp"""
        # add timestamp to the action data
        action_with_timestamp = action_data.copy()
        action_with_timestamp['timestamp'] = datetime.now().isoformat()
        
        self.actions.append(action_with_timestamp)
        
        # limit the history size
        if len(self.actions) > self.max_history:
            self.actions = self.actions[-self.max_history:]
    
    def get_history_string(self) -> str:
        """get formatted history string"""
        if not self.actions:
            return "No previous actions"
        
        history_lines = []
        total_actions = len(self.actions)
        
        for i, action in enumerate(self.actions, 1):
            timestamp = action.get('timestamp', 'N/A')
            screen_analysis = action.get('screen_analysis', 'N/A')
            verification = action.get('previous_action_verification', 'N/A')
            next_action = action.get('next_action', 'N/A')
            grounded_action = action.get('grounded_action', 'N/A')
            
            # format timestamp for display
            time_display = self._format_timestamp(timestamp)
            
            history_lines.append(f"Action {i} (of {total_actions}) - {time_display}:")
            # history_lines.append(f"  Screen Analysis: {screen_analysis}")
            history_lines.append(f"  Verification: {verification}")
            history_lines.append(f"  Next Action: {next_action}")
            history_lines.append(f"  Grounded Action: {grounded_action}")
            history_lines.append("") 
        
        return "\n".join(history_lines)
    
    def _format_timestamp(self, timestamp: str) -> str:
        """format timestamp to a readable format"""
        try:
            if timestamp == 'N/A':
                return 'N/A'
            dt = datetime.fromisoformat(timestamp)
            return dt.strftime("%H:%M:%S")
        except:
            return timestamp
    
    def get_action_count(self) -> int:
        """get the number of actions in history"""
        return len(self.actions)
    
    def clear(self):
        """clear the action history"""
        self.actions = []
    
    def is_at_max_capacity(self) -> bool:
        """judge if the history has reached its maximum capacity"""
        return len(self.actions) >= self.max_history
    
    def get_last_action_summary(self) -> str:
        """get a summary of the last action, formatted for template input
        
        Returns:
            includes screen_analysis, next_action and grounded_action in a formatted string
        """
        if not self.actions:
            return "No previous action"
        
        last_action = self.actions[-1]
        screen_analysis = last_action.get('screen_analysis', 'N/A')
        next_action = last_action.get('next_action', 'N/A')
        grounded_action = last_action.get('grounded_action', 'N/A')
        
        return f"""Previous Action Details:
        - Intended Action: {next_action}
        - Executed Code: {grounded_action}"""


