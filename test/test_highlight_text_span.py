import sys
import os
# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#!/usr/bin/env python3
"""
Test script for highlight_text_span functionality
"""

from grounding import Agent
import time

def test_highlight_text_span():
    """Test the highlight_text_span method"""
    print("Testing highlight_text_span functionality...")
    
    # Create agent instance
    agent = Agent()
    
    print("\n=== Word Document Text Highlighting Test ===")
    print("请确保你已经打开一个Word文档，并且文档中包含 'hello world' 文本")
    print("测试将在5秒后开始，请确保Word文档在前台显示...")
    
    # Give user time to prepare
    for i in range(5, 0, -1):
        print(f"倒计时: {i}秒")
        time.sleep(1)
    
    print("\n开始测试...")
    
    # Test case: Highlighting "hello world" in Word document
    print("\nTest case: 在Word文档中选中 'hello world'")
    starting_phrase = "hello"
    ending_phrase = "world"
    
    try:
        print(f"正在搜索并选中从 '{starting_phrase}' 到 '{ending_phrase}' 的文本...")
        result = agent.highlight_text_span(starting_phrase, ending_phrase)
        
        if result:
            print(f"✓ 成功选中文本: '{starting_phrase}' 到 '{ending_phrase}'")
            print("如果文本被正确选中，测试成功！")
        else:
            print(f"✗ 未能选中文本: '{starting_phrase}' 到 '{ending_phrase}'")
            print("可能的原因:")
            print("  - Word文档中不包含指定的文本")
            print("  - 文本被其他窗口遮挡")
            print("  - OCR识别失败")
    except Exception as e:
        print(f"✗ 文本选中过程中出现错误: {e}")
    
    print("\n=== 安装要求 ===")
    print("1. 安装 Python 包: pip install pytesseract")
    print("2. 安装 Tesseract OCR 软件:")
    print("   Windows:")
    print("   - 下载: https://github.com/UB-Mannheim/tesseract/wiki")
    print("   - 或使用: winget install UB-Mannheim.TesseractOCR")
    print("   - 安装后添加到 PATH 环境变量")
    print("   macOS:")
    print("   - brew install tesseract")
    print("   Linux:")
    print("   - sudo apt install tesseract-ocr (Ubuntu/Debian)")
    print("   - sudo yum install tesseract (CentOS/RHEL)")
    print("3. 打开Word文档并确保包含 'hello world' 文本")
    print("4. Word文档在测试时保持在前台显示")
    print("5. 文本要清晰可见，不被其他元素遮挡")
    print("\n注意: 如果 Tesseract 不在 PATH 中，可能需要手动指定路径:")
    print("pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'")
    
    return True

if __name__ == "__main__":
    test_highlight_text_span()
