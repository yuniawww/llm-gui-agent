import pyautogui
import time
from typing import List, Dict, Any, Optional, Tuple


# Agent action execution functions
class Agent:
    """Agent class that mimics the runner's agent interface but executes real actions"""
    
    @staticmethod
    def click(x: int, y: int, num_clicks: int = 1, button_type: str = 'left', hold_keys: List[str] = None) -> bool:
        """
        Execute click action using pyautogui
        
        Args:
            x: X coordinate
            y: Y coordinate  
            num_clicks: Number of clicks
            button_type: Mouse button type ('left', 'right', 'middle')
            hold_keys: Keys to hold while clicking
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if hold_keys:
                # Hold keys while clicking
                with pyautogui.hold(hold_keys):
                    pyautogui.click(x, y, clicks=num_clicks, button=button_type)
            else:
                pyautogui.click(x, y, clicks=num_clicks, button=button_type)
            
            print(f"Clicked at ({x}, {y}) with {num_clicks} {button_type} clicks")
            return True
            
        except Exception as e:
            print(f"Error in click action: {e}")
            return False
    
    @staticmethod
    def type(x: int, y: int, text: str, overwrite: bool = False, enter: bool = False) -> bool:
        """
        Execute type action using pyautogui
        
        Args:
            x: X coordinate to click first
            y: Y coordinate to click first
            text: Text to type
            overwrite: Whether to clear existing text first
            enter: Whether to press enter after typing
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Click to focus the element first
            pyautogui.click(x, y)
            
            # Give more time for focus to be established
            time.sleep(0.2)
            
            # Clear existing text if overwrite is True
            if overwrite:
                pyautogui.hotkey('ctrl', 'a')  # Select all
                time.sleep(0.1)  # Wait for selection
                pyautogui.press('delete')      # Delete selected text
                time.sleep(0.1)  # Wait for deletion
            
            # Type the text with slower interval to ensure all characters are typed
            print(f"Starting to type: '{text}'")
            pyautogui.write(text, interval=0.05)  # Add small interval between characters
            
            # Small delay to ensure typing is complete
            time.sleep(0.2)
            
            # Press enter if requested
            if enter:
                pyautogui.press('enter')
            
            print(f"Typed '{text}' at ({x}, {y}), overwrite={overwrite}, enter={enter}")
            return True
            
        except Exception as e:
            print(f"Error in type action: {e}")
            return False
    
    @staticmethod
    def scroll(x: int, y: int, clicks: int, shift: bool = False) -> bool:
        """
        Execute scroll action using pyautogui
        
        Args:
            x: X coordinate of scroll location
            y: Y coordinate of scroll location
            clicks: Number of scroll clicks (positive=up, negative=down)
            shift: Whether to use shift+scroll for horizontal scrolling
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if shift:
                # Horizontal scroll with shift
                with pyautogui.hold(['shift']):
                    pyautogui.scroll(clicks, x=x, y=y)
            else:
                # Vertical scroll
                pyautogui.scroll(clicks, x=x, y=y)
            
            direction = "up" if clicks > 0 else "down"
            print(f"Scrolled {abs(clicks)} clicks {direction} at ({x}, {y})")
            return True
            
        except Exception as e:
            print(f"Error in scroll action: {e}")
            return False
    
    @staticmethod
    def drag_and_drop(start_x: int, start_y: int, end_x: int, end_y: int, hold_keys: List[str] = None) -> bool:
        """
        Execute drag and drop action using pyautogui
        
        Args:
            start_x: Starting X coordinate
            start_y: Starting Y coordinate
            end_x: Ending X coordinate
            end_y: Ending Y coordinate
            hold_keys: Keys to hold during drag operation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Move to start position first
            pyautogui.moveTo(start_x, start_y)
            time.sleep(0.1)  # Brief pause
            
            if hold_keys:
                # Hold keys while dragging
                with pyautogui.hold(hold_keys):
                    pyautogui.dragTo(end_x, end_y, duration=1)
            else:
                pyautogui.dragTo(end_x, end_y, duration=1)
            
            print(f"Dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})")
            return True
            
        except Exception as e:
            print(f"Error in drag_and_drop action: {e}")
            return False
    
    @staticmethod
    def done(return_value: Any = None) -> bool:
        """
        Task completion marker
        
        Args:
            return_value: Optional return value
            
        Returns:
            True (always successful)
        """
        print("Task completed successfully")
        return True
    
    @staticmethod
    def fail() -> bool:
        """
        Task failure marker
        
        Returns:
            False (indicates failure)
        """
        print("Task failed")
        return False
    
    @staticmethod
    def switch_applications(app_code: str) -> bool:
        """
        Switch to a different application that is already open
        
        Args:
            app_code: The code name of the application to switch to            
        Returns:
            True if successful, False otherwise
        """
        try:
            import platform
            current_platform = platform.system().lower()
            
            if current_platform == "linux":  # Linux
                # Basic implementation - can be enhanced
                pyautogui.hotkey('alt', 'f2')
                time.sleep(0.5)
                pyautogui.typewrite(app_code)
                pyautogui.press('enter')
                time.sleep(1.0)
            elif current_platform == "windows":  # Windows
                pyautogui.hotkey('win')
                time.sleep(1)
                pyautogui.typewrite(app_code)
                time.sleep(1)
                pyautogui.press('enter')
                time.sleep(2.0)
                pyautogui.hotkey('win')
                time.sleep(1)
                pyautogui.hotkey('win')
                time.sleep(1)
            else:
                print(f"Platform {current_platform} not supported")
                return False
            
            print(f"Switched to application: {app_code}")
            return True
            
        except Exception as e:
            print(f"Error in switch_applications: {e}")
            return False
    
    @staticmethod
    def open(app_or_filename: str) -> bool:
        """
        Open any application or file with name app_or_filename
        
        Args:
            app_or_filename: The name of the application or filename to open
            
        Returns:
            True if successful, False otherwise
        """
        try:
            pyautogui.hotkey('win')
            time.sleep(0.5)
            pyautogui.write(app_or_filename)
            time.sleep(1.0)
            pyautogui.press('enter')
            time.sleep(0.5)
            
            print(f"Opened: {app_or_filename}")
            return True
            
        except Exception as e:
            print(f"Error in open: {e}")
            return False
    
    @staticmethod
    def hotkey(keys: List[str]) -> bool:
        """
        Press a hotkey combination
        
        Args:
            keys: List of keys to press in combination (e.g., ['ctrl', 'c'])
            
        Returns:
            True if successful, False otherwise
        """
        try:
            pyautogui.hotkey(*keys)
            print(f"Pressed hotkey: {'+'.join(keys)}")
            return True
            
        except Exception as e:
            print(f"Error in hotkey: {e}")
            return False
    
    @staticmethod
    def hold_and_press(hold_keys: List[str], press_keys: List[str]) -> bool:
        """
        Hold a list of keys and press a list of keys
        
        Args:
            hold_keys: List of keys to hold
            press_keys: List of keys to press in sequence
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Hold down the keys
            for key in hold_keys:
                pyautogui.keyDown(key)
            
            # Press the keys in sequence
            for key in press_keys:
                pyautogui.press(key)
            
            # Release the held keys
            for key in hold_keys:
                pyautogui.keyUp(key)
            
            print(f"Held {hold_keys} and pressed {press_keys}")
            return True
            
        except Exception as e:
            print(f"Error in hold_and_press: {e}")
            # Make sure to release any held keys on error
            for key in hold_keys:
                try:
                    pyautogui.keyUp(key)
                except:
                    pass
            return False
    
    @staticmethod
    def wait(wait_time: float) -> bool:
        """
        Wait for a specified amount of time
        
        Args:
            wait_time: The amount of time to wait in seconds
            
        Returns:
            True (always successful)
        """
        try:
            time.sleep(wait_time)
            print(f"Waited for {wait_time} seconds")
            return True
            
        except Exception as e:
            print(f"Error in wait: {e}")
            return False
    
    @staticmethod
    def set_cell_values(cell_values: Dict[str, str], app_name: str, sheet_name: str = None) -> bool:
        """
        Set values in spreadsheet cells using appropriate interface
        
        Args:
            cell_values: Dictionary mapping cell addresses to values (e.g., {"A1": "Hello", "B2": "World"})
            app_name: Application name (e.g., "Excel", "Calc", "LibreOffice Calc")
            sheet_name: Optional sheet name (uses active sheet if None)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            app_name_lower = app_name.lower()
            print(f"🔍 Debug: Analyzing app_name='{app_name}' (lowercase: '{app_name_lower}')")
            
            # Check if it's Microsoft Office (Excel)
            # Support both app names and file extensions
            is_excel = (
                any(ms_app in app_name_lower for ms_app in ['excel', 'microsoft', 'office']) or
                app_name_lower.endswith('.xlsx') or
                app_name_lower.endswith('.xls') or
                app_name_lower.endswith('.xlsm') or
                app_name_lower.endswith('.xlsb')
            )
            
            print(f"🔍 Debug: is_excel={is_excel}")
            
            if is_excel:
                # Excel COM interface
                try:
                    import win32com.client
                    
                    # Connect to Excel application
                    excel_app = win32com.client.Dispatch("Excel.Application")
                    
                    # Get active workbook
                    workbook = excel_app.ActiveWorkbook
                    if not workbook:
                        print("No active Excel workbook found")
                        return False
                    
                    # Get worksheet
                    if sheet_name:
                        try:
                            worksheet = workbook.Worksheets(sheet_name)
                        except:
                            print(f"Sheet '{sheet_name}' not found")
                            return False
                    else:
                        worksheet = workbook.ActiveSheet
                    
                    # Set cell values
                    for cell_address, value in cell_values.items():
                        try:
                            worksheet.Range(cell_address).Value = value
                            print(f"Set {cell_address} = '{value}' in Excel")
                        except Exception as e:
                            print(f"Error setting cell {cell_address}: {e}")
                    
                    # Save the workbook
                    try:
                        workbook.Save()
                        print("Excel workbook saved")
                    except Exception as e:
                        print(f"Warning: Could not save workbook: {e}")
                    
                    return True
                    
                except ImportError:
                    print("pywin32 not installed. Install with: pip install pywin32")
                    return False
                except Exception as e:
                    print(f"Error in Excel COM interface: {e}")
                    return False
            
            # Check if it's LibreOffice
            # Support both app names and file extensions
            elif (
                any(libre_app in app_name_lower for libre_app in ['libreoffice', 'libre', 'calc']) or
                app_name_lower.endswith('.ods') or
                app_name_lower.endswith('.fods') or
                app_name_lower.endswith('.sxc')
            ):
                print(f"🔍 Debug: Detected LibreOffice application")
                # LibreOffice UNO interface
                try:
                    # Parse cell address helper function
                    def parse_cell_address(cell_address: str) -> tuple:
                        import re
                        match = re.match(r'^([A-Z]+)(\d+)$', cell_address.upper())
                        if not match:
                            raise ValueError(f"Invalid cell address: {cell_address}")
                        
                        column_str, row_str = match.groups()
                        
                        # Convert column letters to 0-based index
                        column_index = 0
                        for char in column_str:
                            column_index = column_index * 26 + (ord(char) - ord('A') + 1)
                        column_index -= 1  # Convert to 0-based
                        
                        # Convert row to 0-based index
                        row_index = int(row_str) - 1
                        
                        return column_index, row_index
                    
                    import uno
                    from com.sun.star.connection import NoConnectException
                    
                    # Connect to LibreOffice
                    local_context = uno.getComponentContext()
                    resolver = local_context.ServiceManager.createInstanceWithContext(
                        "com.sun.star.bridge.UnoUrlResolver", local_context
                    )
                    
                    try:
                        # Try to connect to running LibreOffice instance
                        context = resolver.resolve("uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext")
                    except NoConnectException:
                        print("LibreOffice not running or UNO connection not available")
                        print("Please start LibreOffice with: soffice --accept='socket,host=localhost,port=2002;urp;'")
                        return False
                    
                    # Get desktop
                    desktop = context.ServiceManager.createInstanceWithContext(
                        "com.sun.star.frame.Desktop", context
                    )
                    
                    # Get current document
                    document = desktop.getCurrentComponent()
                    if not document:
                        print("No active LibreOffice document found")
                        return False
                    
                    # Check if it's a Calc document
                    if not hasattr(document, 'getSheets'):
                        print("Active document is not a Calc spreadsheet")
                        return False
                    
                    # Get worksheet
                    sheets = document.getSheets()
                    if sheet_name:
                        try:
                            worksheet = sheets.getByName(sheet_name)
                        except:
                            print(f"Sheet '{sheet_name}' not found")
                            return False
                    else:
                        # Get active sheet
                        worksheet = document.getCurrentController().getActiveSheet()
                    
                    # Set cell values
                    for cell_address, value in cell_values.items():
                        try:
                            # Parse cell address (e.g., "A1" -> column 0, row 0)
                            col, row = parse_cell_address(cell_address)
                            
                            # Get cell and set value
                            cell = worksheet.getCellByPosition(col, row)
                            
                            # Try to convert to number if possible, otherwise use string
                            try:
                                numeric_value = float(value)
                                cell.setValue(numeric_value)
                            except ValueError:
                                cell.setString(str(value))
                            
                            print(f"Set {cell_address} = '{value}' in LibreOffice Calc")
                            
                        except Exception as e:
                            print(f"Error setting cell {cell_address}: {e}")
                    
                    # Save the document
                    try:
                        document.store()
                        print("LibreOffice document saved")
                    except Exception as e:
                        print(f"Warning: Could not save document: {e}")
                    
                    return True
                    
                except ImportError as import_error:
                    print(f"LibreOffice UNO Python bindings not available: {import_error}")
                    print("Make sure LibreOffice is installed and UNO is accessible")
                    return False
                except Exception as e:
                    print(f"Error in LibreOffice UNO interface: {e}")
                    return False
            
            else:
                print(f"🔍 Debug: Application not recognized as Excel or LibreOffice")
                print(f"❌ Unsupported application: {app_name}")
                print(f"   Supported formats:")
                print(f"   - Excel: files ending with .xlsx, .xls, .xlsm, .xlsb")
                print(f"   - LibreOffice: files ending with .ods, .fods, .sxc")
                print(f"   - App names containing: excel, microsoft, office, libreoffice, libre, calc")
                return False
                
        except Exception as e:
            print(f"Error in set_cell_values: {e}")
            return False
    
    @staticmethod
    def highlight_text_span(starting_phrase: str, ending_phrase: str) -> bool:
        """
        Highlight text span from starting_phrase to ending_phrase using OCR and pyautogui
        Enhanced with better text matching strategies
        
        Args:
            starting_phrase: The text phrase to start highlighting from
            ending_phrase: The text phrase to end highlighting at
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import pytesseract
            from PIL import Image
            import pyautogui
            
            # Take screenshot
            screenshot = pyautogui.screenshot()
            
            # Use OCR to extract text with bounding boxes
            ocr_data = pytesseract.image_to_data(screenshot, output_type=pytesseract.Output.DICT)
            
            # Get all text words with their positions
            words = []
            for i, text in enumerate(ocr_data['text']):
                if text.strip():  # Skip empty text
                    words.append({
                        'text': text.strip(),
                        'x': ocr_data['left'][i],
                        'y': ocr_data['top'][i],
                        'width': ocr_data['width'][i],
                        'height': ocr_data['height'][i]
                    })
            
            print(f"OCR found {len(words)} words on screen")
            if len(words) > 0:
                print(f"Sample words: {[w['text'] for w in words[:5]]}")  # Show first 5 words for debugging
            
            # Find starting phrase position with multiple strategies
            start_result = Agent._find_text_position(starting_phrase, words, is_start=True)
            end_result = Agent._find_text_position(ending_phrase, words, is_start=False)
            
            if not start_result:
                print(f"Starting phrase '{starting_phrase}' not found")
                print(f"Available words: {[w['text'] for w in words[:10]]}")  # Show more context
                return False
            if not end_result:
                print(f"Ending phrase '{ending_phrase}' not found")
                print(f"Available words: {[w['text'] for w in words[:10]]}")  # Show more context
                return False
            
            start_x, start_y = start_result
            end_x, end_y = end_result
            
            # Validate coordinates
            if start_x < 0 or start_y < 0 or end_x < 0 or end_y < 0:
                print(f"Invalid coordinates: start=({start_x}, {start_y}), end=({end_x}, {end_y})")
                return False
            
            # Generate and execute the drag selection
            print(f"Highlighting text from ({start_x}, {start_y}) to ({end_x}, {end_y})")
            print(f"Generated commands: pyautogui.moveTo({start_x}, {start_y}); pyautogui.dragTo({end_x}, {end_y})")
            
            # Move to starting position and drag to end position
            pyautogui.moveTo(start_x, start_y)
            time.sleep(0.1)
            pyautogui.dragTo(end_x, end_y, duration=0.5)
            
            return True
            
        except ImportError:
            print("pytesseract not installed. Install with: pip install pytesseract")
            print("Also ensure Tesseract OCR is installed on your system")
            return False
        except Exception as e:
            print(f"Error in highlight_text_span: {e}")
            return False
    
    @staticmethod
    def _find_text_position(phrase: str, words: List[Dict], is_start: bool = True) -> Optional[Tuple[int, int]]:
        """
        Find text position using multiple matching strategies (simple but effective)
        
        Args:
            phrase: The phrase to find
            words: List of word dictionaries with position information
            is_start: True for start position (left edge), False for end position (right edge)
            
        Returns:
            Tuple of (x, y) coordinates or None if not found
        """
        phrase_lower = phrase.lower().strip()
        
        # Strategy 1: Exact match (highest priority)
        for word in words:
            if word['text'].lower() == phrase_lower:
                x = word['x'] if is_start else word['x'] + word['width']
                y = word['y'] + word['height'] // 2
                print(f"Exact match: '{phrase}' -> '{word['text']}' at ({x}, {y})")
                return (x, y)
        
        # Strategy 2: Word starts with phrase
        for word in words:
            if word['text'].lower().startswith(phrase_lower):
                x = word['x'] if is_start else word['x'] + word['width']
                y = word['y'] + word['height'] // 2
                print(f"Start match: '{phrase}' -> '{word['text']}' at ({x}, {y})")
                return (x, y)
        
        # Strategy 3: Phrase starts with word (reverse check)
        for word in words:
            if phrase_lower.startswith(word['text'].lower()) and len(word['text']) >= 2:
                x = word['x'] if is_start else word['x'] + word['width']
                y = word['y'] + word['height'] // 2
                print(f"Phrase-start match: '{phrase}' -> '{word['text']}' at ({x}, {y})")
                return (x, y)
        
        # Strategy 4: Contains match (only if phrase is long enough to avoid false positives)
        if len(phrase_lower) >= 4:  # Only for longer phrases
            for word in words:
                if phrase_lower in word['text'].lower() or word['text'].lower() in phrase_lower:
                    x = word['x'] if is_start else word['x'] + word['width']
                    y = word['y'] + word['height'] // 2
                    print(f"Contains match: '{phrase}' -> '{word['text']}' at ({x}, {y})")
                    return (x, y)
        
        return None