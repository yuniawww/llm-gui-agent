#!/usr/bin/env python3
"""
Parameter Parser Utility
Safely parse function parameters from strings, handling both positional and named parameters
"""

import ast
import re
from typing import List, Any


def safe_parse_params(params_str: str) -> List[Any]:
    """
    Safely parse function parameters from string, handling both positional and named parameters
    Uses ast.literal_eval for safe evaluation without code execution risks
    
    Args:
        params_str: Parameter string like "150, 200" or "text='hello', x=997, y=688" 
        
    Returns:
        List of parsed parameters (for positional) or [dict] (for named parameters)
    """
    try:
        # Handle empty parameters
        if not params_str.strip():
            return []
        
        # Check if this contains named parameters (= outside quotes)
        has_named_params = _has_named_parameters(params_str)
        
        if has_named_params:
            try:
                # Parse named parameters into a dictionary
                param_dict = {}
                
                # Split by comma, but be careful with quoted strings
                params = _smart_split_params(params_str)
                
                for param in params:
                    param = param.strip()
                    if '=' in param:
                        # Find the first '=' that's not inside quotes
                        eq_index = _find_equal_outside_quotes(param)
                        if eq_index != -1:
                            key = param[:eq_index].strip()
                            value = param[eq_index + 1:].strip()
                            
                            # Parse the value safely
                            parsed_value = _parse_single_value(value)
                            param_dict[key] = parsed_value
                
                # Smart conversion: convert named parameters to positional order
                converted = _convert_named_dict_to_positional(param_dict)
                if converted is not None:
                    return converted
                        
                return [param_dict]  # Fallback: return as single dict parameter
            except Exception as e:
                print(f"Error parsing named parameters: {e}")
                return []
        else:
            # Handle positional parameters using ast.literal_eval
            try:
                # First try direct ast.literal_eval with bracket wrapper
                return ast.literal_eval(f"[{params_str}]")
            except (ValueError, SyntaxError):
                # Fallback: parse each parameter individually 
                try:
                    result = []
                    params = _smart_split_params(params_str)
                    for param in params:
                        param = param.strip()
                        parsed_value = _parse_single_value(param)
                        result.append(parsed_value)
                    return result
                except Exception as e:
                    print(f"Error parsing positional parameters: {e}")
                    return []
            
    except Exception as e:
        print(f"Error parsing parameters '{params_str}': {e}")
        return []


def _has_named_parameters(params_str: str) -> bool:
    """
    Check if the parameter string contains named parameters (= outside quotes)
    
    Args:
        params_str: Parameter string to check
        
    Returns:
        True if named parameters are found, False otherwise
    """
    in_quotes = False
    quote_char = None
    
    for char in params_str:
        if char in ('"', "'") and not in_quotes:
            in_quotes = True
            quote_char = char
        elif char == quote_char and in_quotes:
            in_quotes = False
            quote_char = None
        elif char == '=' and not in_quotes:
            return True
    
    return False


def _find_equal_outside_quotes(param_str: str) -> int:
    """
    Find the first '=' character that's not inside quotes
    
    Args:
        param_str: Parameter string to search
        
    Returns:
        Index of first '=' outside quotes, or -1 if not found
    """
    in_quotes = False
    quote_char = None
    
    for i, char in enumerate(param_str):
        if char in ('"', "'") and not in_quotes:
            in_quotes = True
            quote_char = char
        elif char == quote_char and in_quotes:
            in_quotes = False
            quote_char = None
        elif char == '=' and not in_quotes:
            return i
    
    return -1


def _smart_split_params(params_str: str) -> List[str]:
    """
    Smart parameter splitting that respects quoted strings
    
    Args:
        params_str: Parameter string to split
        
    Returns:
        List of parameter strings
    """
    try:
        # For simple cases without quotes, use regular split
        if '"' not in params_str and "'" not in params_str:
            return [p.strip() for p in params_str.split(',')]
        
        # For complex cases with quotes, use a simple state machine
        params = []
        current_param = ""
        in_quotes = False
        quote_char = None
        
        i = 0
        while i < len(params_str):
            char = params_str[i]
            
            if char in ('"', "'") and not in_quotes:
                # Start of quoted string
                in_quotes = True
                quote_char = char
                current_param += char
            elif char == quote_char and in_quotes:
                # End of quoted string
                in_quotes = False
                quote_char = None
                current_param += char
            elif char == ',' and not in_quotes:
                # Parameter separator outside quotes
                params.append(current_param.strip())
                current_param = ""
            else:
                # Regular character
                current_param += char
            
            i += 1
        
        # Add the last parameter
        if current_param.strip():
            params.append(current_param.strip())
        
        return params
        
    except Exception:
        # Fallback to simple split
        return [p.strip() for p in params_str.split(',')]


def _parse_single_value(value_str: str) -> Any:
    """
    Parse a single parameter value safely
    
    Args:
        value_str: String representation of a value
        
    Returns:
        Parsed value (int, float, string, bool, etc.)
    """
    value_str = value_str.strip()
    
    # Handle empty string
    if not value_str:
        return ""
    
    # Try ast.literal_eval first (handles numbers, strings, bools, None, lists, tuples, dicts)
    try:
        return ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
        pass
    
    # Handle unquoted strings (fallback)
    return value_str


def _convert_named_dict_to_positional(param_dict: dict) -> List[Any]:
    """
    Convert named parameters dictionary to positional parameters list
    Automatically detects function type based on parameter names
    
    Args:
        param_dict: Dictionary of named parameters
        
    Returns:
        List of positional parameters or None if conversion not supported
    """
    try:
        # Detect function type based on parameter names
        param_keys = set(param_dict.keys())
        
        if 'x' in param_keys and 'y' in param_keys and 'text' in param_keys:
            # type(x, y, text, overwrite=False, enter=False)
            x = param_dict.get('x', 0)
            y = param_dict.get('y', 0)
            text = param_dict.get('text', '')
            overwrite = param_dict.get('overwrite', False)
            enter = param_dict.get('enter', False)
            print(f"Converting type params: x={x}, y={y}, text='{text}', overwrite={overwrite}, enter={enter}")
            return [x, y, text, overwrite, enter]
            
        elif 'x' in param_keys and 'y' in param_keys and 'num_clicks' in param_keys:
            # click(x, y, num_clicks=1, button_type='left', hold_keys=[])
            x = param_dict.get('x', 0)
            y = param_dict.get('y', 0)
            num_clicks = param_dict.get('num_clicks', 1)
            button_type = param_dict.get('button_type', 'left')
            hold_keys = param_dict.get('hold_keys', [])
            return [x, y, num_clicks, button_type, hold_keys]
            
        elif 'x' in param_keys and 'y' in param_keys and 'clicks' in param_keys:
            # scroll(x, y, clicks, shift=False)
            x = param_dict.get('x', 0)
            y = param_dict.get('y', 0)
            clicks = param_dict.get('clicks', 3)
            shift = param_dict.get('shift', False)
            return [x, y, clicks, shift]
            
        elif 'start_x' in param_keys and 'start_y' in param_keys and 'end_x' in param_keys and 'end_y' in param_keys:
            # drag_and_drop(start_x, start_y, end_x, end_y, hold_keys=[])
            start_x = param_dict.get('start_x', 0)
            start_y = param_dict.get('start_y', 0)
            end_x = param_dict.get('end_x', 0)
            end_y = param_dict.get('end_y', 0)
            hold_keys = param_dict.get('hold_keys', [])
            return [start_x, start_y, end_x, end_y, hold_keys]
            
        # If no pattern matches, return None to use fallback
        return None
        
    except Exception as e:
        print(f"Error converting named dict to positional: {e}")
        return None
