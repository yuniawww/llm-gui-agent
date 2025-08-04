# utils/screenshot.py
import pyautogui
import os
from datetime import datetime
import pygetwindow as gw
import time

def is_cli_window(title: str) -> bool:
    """judge if the window title indicates a CLI (Command Line Interface)"""
    keywords = [
        'cmd', 'powershell', 'terminal','command prompt', 'anaconda', 'prompt', 'shell'
    ]
    title = title.lower()
    return any(kw in title for kw in keywords)

def cleanup_old_screenshots(save_dir="screenshots", max_screenshots=30):
    """clean up old screenshots to keep only the latest `max_screenshots` files"""
    if not os.path.exists(save_dir):
        return
    
    # get all screenshot files in the directory
    screenshot_files = []
    for filename in os.listdir(save_dir):
        if filename.startswith("screenshot_") and filename.endswith(".png"):
            filepath = os.path.join(save_dir, filename)
            # get the last modified time
            mtime = os.path.getmtime(filepath)
            screenshot_files.append((mtime, filepath, filename))
    
    # sort files by last modified time (newest first)
    screenshot_files.sort(reverse=True)
    
    # if there are more files than the max limit, delete the oldest ones
    if len(screenshot_files) > max_screenshots:
        files_to_delete = screenshot_files[max_screenshots:]
        for _, filepath, filename in files_to_delete:
            try:
                os.remove(filepath)
                print(f"[Info] Cleaned up old screenshot: {filename}")
            except Exception as e:
                print(f"[Warning] Could not delete {filename}: {e}")

def take_screenshot(save_dir="screenshots", max_screenshots=30):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{timestamp}.png"
    filepath = os.path.join(save_dir, filename)

    try:
        current_window = gw.getActiveWindow()
        minimized = False

        if current_window and is_cli_window(current_window.title):
            current_window.minimize()
            minimized = True
            time.sleep(0.6)  

    except Exception as e:
        print(f"[Warning] Could not minimize CLI window: {e}")

    # Take the screenshot
    screenshot = pyautogui.screenshot()
    screenshot.save(filepath)
    print(f"[Info] Screenshot saved: {filepath}")
    
    # cleanup old screenshots
    cleanup_old_screenshots(save_dir, max_screenshots)

    try:
        if current_window and minimized:
            current_window.restore()
            time.sleep(0.5)
    except Exception as e:
        print(f"[Warning] Could not restore CLI window: {e}")

    return filepath
