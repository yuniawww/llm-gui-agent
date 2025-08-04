#!/usr/bin/env python3
"""
Accessibility Tree Utilities - Refactored and Optimized
Stable cross-platform accessibility tree extraction and coordinate correction tool
"""

import platform
import time
from typing import Dict, Any, Optional, List, Tuple

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Tree structure limits
MAX_TREE_DEPTH = 15  # Increased from 9 to capture deeper UI hierarchies
MAX_CHILDREN_PER_NODE = 100  # Increased to capture more child elements
MAX_PROCESSED_CHILDREN = 100  # Increased to process more children

# String length limits
MAX_TEXT_LENGTH = 200
MAX_VALUE_LENGTH = 500

# Timing settings
CONTROL_TIMEOUT = 0.1
WINDOW_DELAY = 0.1

# Size thresholds
MIN_CLICKABLE_SIZE = 1

# Clickable control types
CLICKABLE_TYPES = [
    "button", "hyperlink", "menuitem", "listitem", "treeitem", "tabitem",
    "text", "edit", "document", "pane", "groupcontrol", "window",
    "scrollbar", "thumb", "combobox", "checkbox", "radiobutton"
]

# =============================================================================
# PLATFORM DETECTION AND IMPORTS
# =============================================================================

WINDOWS_AVAILABLE = False
LINUX_AVAILABLE = False

if platform.system() == "Windows":
    try:
        import win32gui
        import win32con
        import win32api
        import ctypes
        from uiautomation import Control, WindowControl, GetForegroundWindow
        WINDOWS_AVAILABLE = True
    except ImportError:
        print("Warning: Windows UI Automation libraries not available")
        
elif platform.system() == "Linux":
    try:
        import subprocess
        LINUX_AVAILABLE = True
    except ImportError:
        print("Warning: Linux accessibility libraries not available")

# =============================================================================
# DPI COORDINATE CORRECTION
# =============================================================================

def get_dpi_scale_factor() -> float:
    """Get DPI scale factor for coordinate correction"""
    if not WINDOWS_AVAILABLE:
        return 1.0
        
    try:
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAwarenessContext()
        
        hdc = user32.GetDC(0)
        dpi_x = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
        user32.ReleaseDC(0, hdc)
        
        scale_factor = dpi_x / 96.0
        return scale_factor
    except Exception:
        return 1.0

def correct_coordinates_for_actual_click(x: int, y: int) -> Tuple[int, int]:
    """
    Direct coordinate pass-through for DPI-aware applications.
    
    After testing, confirmed that UIAutomation returns physical coordinates
    when process is DPI-aware (SetProcessDPIAware() is called).
    
    Args:
        x, y: Physical coordinates from accessibility tree
        
    Returns:
        Same coordinates for actual clicking (no conversion needed)
        
    Note: No DPI conversion required in DPI-aware mode
    """
    # In DPI-aware mode, UIAutomation already returns physical coordinates
    # No conversion needed - coordinates can be used directly for clicking
    return x, y

# =============================================================================
# CORE EXTRACTOR CLASS
# =============================================================================

class AccessibilityTreeExtractor:
    """Accessibility tree extractor - simplified and optimized"""
    
    def get_foreground_window_info(self) -> Tuple[Optional[int], Optional[str]]:
        """Get foreground window information"""
        if WINDOWS_AVAILABLE:
            return self._get_windows_foreground_window()
        elif LINUX_AVAILABLE:
            return self._get_linux_foreground_window()
        return None, None
    
    def _get_windows_foreground_window(self) -> Tuple[Optional[int], Optional[str]]:
        """Get Windows foreground window"""
        try:
            hwnd = win32gui.GetForegroundWindow()
            if not hwnd or not win32gui.IsWindow(hwnd):
                return None, None
            
            title = win32gui.GetWindowText(hwnd)
            return hwnd, title
        except Exception as e:
            print(f"Failed to get Windows foreground window: {e}")
            return None, None
    
    def _get_linux_foreground_window(self) -> Tuple[Optional[int], Optional[str]]:
        """Get Linux foreground window"""
        try:
            result = subprocess.run(['xdotool', 'getactivewindow'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                window_id = int(result.stdout.strip())
                title_result = subprocess.run(['xdotool', 'getwindowname', str(window_id)], 
                                            capture_output=True, text=True)
                title = title_result.stdout.strip() if title_result.returncode == 0 else "Unknown"
                return window_id, title
        except Exception as e:
            print(f"Failed to get Linux foreground window: {e}")
        return None, None
    
    def get_window_accessibility_tree(self, window_handle: int) -> Optional[Dict]:
        """Get window accessibility tree"""
        if not self._validate_window_handle(window_handle):
            return None
        
        if WINDOWS_AVAILABLE:
            return self._get_windows_accessibility_tree(window_handle)
        elif LINUX_AVAILABLE:
            return self._get_linux_accessibility_tree(window_handle)
        return None
    
    def _validate_window_handle(self, window_handle: int) -> bool:
        """Validate window handle"""
        if not WINDOWS_AVAILABLE:
            return True
        
        try:
            return win32gui.IsWindow(window_handle)
        except Exception:
            return False
    
    def _get_windows_accessibility_tree(self, window_handle: int) -> Optional[Dict]:
        """Get Windows accessibility tree with improved stability"""
        try:
            # Try to get window control
            window_control = None
            try:
                window_control = WindowControl(Handle=window_handle)
                if not window_control.Exists():
                    window_control = None
            except Exception:
                pass
            
            # Fallback to foreground window
            if not window_control:
                try:
                    window_control = GetForegroundWindow()
                    if not window_control or not window_control.Exists():
                        window_control = None
                except Exception:
                    pass
            
            if not window_control:
                return None
            
            # Wait a bit longer for UI to stabilize (especially for deeper trees)
            time.sleep(WINDOW_DELAY * 2)
            
            return self._build_control_tree(window_control)
            
        except Exception as e:
            print(f"Failed to get Windows accessibility tree: {e}")
            return None
    
    def _get_linux_accessibility_tree(self, window_handle: int) -> Optional[Dict]:
        """Get Linux accessibility tree (basic implementation)"""
        return {
            "control_type": "Window",
            "name": "Linux Window",
            "automation_id": str(window_handle),
            "class_name": "Unknown",
            "bounding_rect": {"left": 0, "top": 0, "right": 800, "bottom": 600, "width": 800, "height": 600},
            "is_enabled": True,
            "is_visible": True,
            "children": []
        }
    
    def _build_control_tree(self, control: Control, depth: int = 0) -> Dict:
        """Build control tree recursively"""
        if depth >= MAX_TREE_DEPTH:
            return {}
        
        try:
            # Get basic information
            info = self._get_control_info(control)
            
            # Process child controls
            try:
                children = control.GetChildren()
                if children and len(children) <= MAX_CHILDREN_PER_NODE:
                    processed = 0
                    for child in children:
                        if processed >= MAX_PROCESSED_CHILDREN:
                            break
                        
                        if self._is_valid_child(child):
                            child_info = self._build_control_tree(child, depth + 1)
                            if child_info:
                                info["children"].append(child_info)
                                processed += 1
            except Exception:
                pass
            
            return info
            
        except Exception:
            return {}
    
    def _get_control_info(self, control: Control) -> Dict:
        """Get control information safely"""
        info = {
            "control_type": "Unknown",
            "name": "",
            "automation_id": "",
            "class_name": "",
            "bounding_rect": {"left": 0, "top": 0, "right": 0, "bottom": 0, "width": 0, "height": 0},
            "is_enabled": True,
            "is_visible": True,
            "children": []
        }
        
        # Safely get attributes
        try:
            info["control_type"] = str(getattr(control, 'ControlTypeName', 'Unknown'))
        except Exception:
            pass
        
        try:
            name = getattr(control, 'Name', '')
            if name:
                info["name"] = str(name)[:MAX_TEXT_LENGTH]
        except Exception:
            pass
        
        try:
            auto_id = getattr(control, 'AutomationId', '')
            if auto_id:
                info["automation_id"] = str(auto_id)[:MAX_TEXT_LENGTH]
        except Exception:
            pass
        
        try:
            class_name = getattr(control, 'ClassName', '')
            if class_name:
                info["class_name"] = str(class_name)[:MAX_TEXT_LENGTH]
        except Exception:
            pass
        
        try:
            rect = control.BoundingRectangle
            if rect:
                info["bounding_rect"] = {
                    "left": int(rect.left),
                    "top": int(rect.top),
                    "right": int(rect.right),
                    "bottom": int(rect.bottom),
                    "width": int(rect.right - rect.left),
                    "height": int(rect.bottom - rect.top)
                }
        except Exception:
            pass
        
        try:
            info["is_enabled"] = bool(getattr(control, 'IsEnabled', True))
        except Exception:
            pass
        
        try:
            info["is_visible"] = bool(getattr(control, 'IsVisible', True))
        except Exception:
            pass
        
        return info
    
    def _is_valid_child(self, child: Control) -> bool:
        """Check if child control is valid - more permissive for deeper tree capture"""
        try:
            # Basic existence check
            if not hasattr(child, 'ControlTypeName'):
                return False
            
            # More permissive size check - allow very small elements
            try:
                rect = child.BoundingRectangle
                # Allow elements with 0 size (some containers may have no visual bounds)
                # Only reject elements with clearly invalid coordinates
                if rect.left < -10000 or rect.top < -10000 or rect.right > 20000 or rect.bottom > 20000:
                    return False
            except Exception:
                # If we can't get bounds, still include the element
                pass
            
            return True
        except Exception:
            return False

# =============================================================================
# FORMATTER CLASS
# =============================================================================

class AccessibilityTreeFormatter:
    """Accessibility tree formatter - simplified and optimized"""
    
    @staticmethod
    def extract_clickable_elements(tree_data: Dict, apply_dpi_correction: bool = True) -> List[Dict]:
        """Extract clickable elements from tree"""
        elements = []
        
        def extract_recursive(control: Dict):
            control_type = control.get("control_type", "").lower()
            name = control.get("name", "").strip()
            rect = control.get("bounding_rect", {})
            
            # Check if clickable
            is_clickable = (
                any(ct in control_type for ct in CLICKABLE_TYPES) or
                name or
                control.get("automation_id", "")
            )
            
            if is_clickable and rect.get("width", 0) > MIN_CLICKABLE_SIZE and rect.get("height", 0) > MIN_CLICKABLE_SIZE:
                # Calculate center point
                center_x = rect.get("left", 0) + rect.get("width", 0) // 2
                center_y = rect.get("top", 0) + rect.get("height", 0) // 2
                
                # Apply DPI correction (note: no actual conversion needed in DPI-aware mode)
                if apply_dpi_correction:
                    corrected_x, corrected_y = correct_coordinates_for_actual_click(center_x, center_y)
                    # Mark as DPI-aware processed (even though no conversion happens)
                    dpi_corrected = True
                else:
                    corrected_x, corrected_y = center_x, center_y
                    dpi_corrected = False
                
                elements.append({
                    "name": name or f"<{control_type}>",
                    "type": control_type,
                    "center_x": corrected_x,
                    "center_y": corrected_y,
                    "original_center_x": center_x,
                    "original_center_y": center_y,
                    "rect": rect,
                    "automation_id": control.get("automation_id", ""),
                    "dpi_corrected": dpi_corrected
                })
            
            # Process children recursively
            for child in control.get("children", []):
                extract_recursive(child)
        
        if tree_data:
            extract_recursive(tree_data)
        
        return elements
    
    @staticmethod
    def format_clickable_elements_for_llm(tree_data: Dict, window_title: str = "", apply_dpi_correction: bool = True) -> str:
        """Format clickable elements for LLM consumption"""
        if not tree_data:
            return "No clickable elements available."
        
        elements = AccessibilityTreeFormatter.extract_clickable_elements(tree_data, apply_dpi_correction)
        
        if not elements:
            return f"=== Clickable Elements for Window: {window_title} ===\nNo clickable elements found."
        
        lines = [
            f"=== Clickable Elements for Window: {window_title} ===",
            f"Found {len(elements)} clickable elements:",
            ""
        ]
        
        for i, elem in enumerate(elements, 1):
            name = elem['name']
            elem_type = elem['type']
            center_x, center_y = elem['center_x'], elem['center_y']
            rect = elem['rect']
            auto_id = elem['automation_id']
            auto_id_info = f" [ID: {auto_id}]" if auto_id else ""
            
            line = f"{i:2d}. {name} ({elem_type}) [Center: ({center_x}, {center_y})] [{size_info}]{auto_id_info}{dpi_info}"
            lines.append(line)
        
        return "\n".join(lines)

# =============================================================================
# ACCESSIBILITY GROUNDER CLASS
# =============================================================================

class AccessibilityGrounder:
    """专门处理accessibility tree grounding的坐标验证和解析"""
    
    def __init__(self):
        self.tree_formatter = AccessibilityTreeFormatter()
        # Initialize LLM for coordinate extraction
        try:
            import google.generativeai as genai
            from config import GEMINI_API_KEY, GEMINI_MODEL_NAME
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        except ImportError:
            print("Warning: Google Generative AI not available for coordinate extraction")
            self.model = None
    
    def validate_accessibility_coordinates(self, grounded_action: str, tree_data: Any) -> str:
        """
        Validate that coordinates in grounded action exist in accessibility tree
        Uses LLM to extract coordinates from action string, then validates locally
        
        Args:
            grounded_action: Action with coordinates (any format)
            tree_data: Accessibility tree data
            
        Returns:
            Validated action or agent.fail() if coordinates not found
        """
        try:
            if not grounded_action.startswith('agent.') or grounded_action in ['agent.done()', 'agent.fail()']:
                return grounded_action
            
            # Use LLM to extract coordinates from action
            coords = self._extract_coordinates_with_llm(grounded_action)
            if not coords:
                print("LLM failed to extract coordinates from action")
                return "agent.fail()"
            
            print(f"LLM extracted coordinates: {coords}")
            
            # Get all valid coordinates from tree data
            valid_coords = self.get_valid_coordinates_from_tree(tree_data)
            
            if not valid_coords:
                print("No valid coordinates found in accessibility tree, allowing action to proceed")
                return grounded_action
            
            # Check if action coordinates are in valid set (with tolerance for small differences)
            tolerance = 5  # Allow 5 pixel tolerance for coordinate matching
            
            for coord_pair in coords:
                action_x, action_y = coord_pair
                found_match = False
                
                # Check for exact match first
                if coord_pair in valid_coords:
                    found_match = True
                else:
                    # Check for match within tolerance
                    for valid_x, valid_y in valid_coords:
                        if (abs(action_x - valid_x) <= tolerance and 
                            abs(action_y - valid_y) <= tolerance):
                            found_match = True
                            print(f"Coordinate {coord_pair} matched with tolerance to ({valid_x}, {valid_y})")
                            break
                
                if not found_match:
                    print(f"Coordinate {coord_pair} not found in accessibility tree (checked {len(valid_coords)} valid coordinates)")
                    # For debugging: show nearby coordinates
                    nearby_coords = [(x, y) for x, y in valid_coords 
                                   if abs(x - action_x) <= 50 and abs(y - action_y) <= 50]
                    if nearby_coords:
                        print(f"Nearby coordinates within 50px: {nearby_coords[:5]}")
                    return "agent.fail()"
            
            print(f"All coordinates validated successfully: {coords}")
            return grounded_action
            
        except Exception as e:
            print(f"Error validating coordinates: {e}")
            return "agent.fail()"
    
    def _extract_coordinates_with_llm(self, grounded_action: str) -> List[Tuple[int, int]]:
        """
        Use LLM to extract coordinates from grounded action string
        
        Args:
            grounded_action: Action string in any format
            
        Returns:
            List of coordinate tuples
        """
        if not self.model:
            print("LLM model not available for coordinate extraction")
            return []
        
        try:
            prompt = f"""
Extract the x,y coordinates from this action string. The action may be in various formats:
- Positional: agent.click(997, 688, 1, 'left', [])
- Named: agent.type(text='hello', x=997, y=688, overwrite=False)
- Mixed: agent.scroll(x=997, y=688, clicks=3)

Action: {grounded_action}

For each coordinate pair found, output in this exact format:
COORDINATES: (x, y)

If it's a drag_and_drop action with start and end coordinates, output:
COORDINATES: (start_x, start_y)
COORDINATES: (end_x, end_y)

If no coordinates found, output:
NO_COORDINATES

Only output the coordinate lines, nothing else.
"""
            
            response = self.model.generate_content(prompt)
            if not response or not response.text:
                print("LLM returned empty response for coordinate extraction")
                return []
            
            # Parse LLM response to extract coordinates
            coords = []
            lines = response.text.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if line.startswith('COORDINATES:'):
                    # Extract coordinates from line like "COORDINATES: (997, 688)"
                    try:
                        coord_str = line.split('COORDINATES:')[1].strip()
                        if coord_str.startswith('(') and coord_str.endswith(')'):
                            coord_str = coord_str[1:-1]  # Remove parentheses
                            x_str, y_str = coord_str.split(',')
                            x, y = int(x_str.strip()), int(y_str.strip())
                            coords.append((x, y))
                    except Exception as e:
                        print(f"Error parsing coordinate line '{line}': {e}")
                elif line == 'NO_COORDINATES':
                    print("LLM reported no coordinates found")
                    return []
            
            return coords
            
        except Exception as e:
            print(f"Error using LLM for coordinate extraction: {e}")
            return []
    
    def get_valid_coordinates_from_tree(self, tree_data: Any) -> set:
        """Get all valid center coordinates from accessibility tree"""
        valid_coords = set()
        try:
            # Use AccessibilityTreeFormatter to extract clickable elements directly
            # This is more reliable than parsing formatted strings
            clickable_elements = self.tree_formatter.extract_clickable_elements(tree_data, apply_dpi_correction=True)
            
            # Extract center coordinates from clickable elements
            for element in clickable_elements:
                center_x = element.get('center_x')
                center_y = element.get('center_y')
                
                if center_x is not None and center_y is not None:
                    valid_coords.add((center_x, center_y))
                    
            print(f"Extracted {len(valid_coords)} valid coordinates from accessibility tree")
            
        except Exception as e:
            print(f"Error getting valid coordinates from tree: {e}")
        
        return valid_coords

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_current_window_tree() -> Tuple[Optional[Dict], Optional[str]]:
    """Get current window tree"""
    extractor = AccessibilityTreeExtractor()
    handle, title = extractor.get_foreground_window_info()
    if handle:
        tree = extractor.get_window_accessibility_tree(handle)
        return tree, title
    return None, None

def get_formatted_current_window_tree(apply_dpi_correction: bool = True) -> str:
    """Get formatted current window tree"""
    tree, title = get_current_window_tree()
    if tree:
        formatter = AccessibilityTreeFormatter()
        return formatter.format_clickable_elements_for_llm(tree, title or "Unknown Window", apply_dpi_correction)
    return "No clickable elements available."
