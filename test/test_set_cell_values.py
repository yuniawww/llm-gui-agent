#!/usr/bin/env python3
"""
Test set_cell_values functionality
"""

from grounding import Agent, take_action
import sys
import os
# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_set_cell_values():
    """Test set_cell_values function"""
    
    print("=" * 60)
    print("🧪 Testing set_cell_values functionality")
    print("=" * 60)
    
    # Test data
    cell_values = {
        "A1": "Name",
        "B1": "Age", 
        "C1": "City",
        "A2": "John",
        "B2": "25",
        "C2": "New York",
        "A3": "Alice",
        "B3": "30",
        "C3": "London"
    }
    
    print("📋 Test data:")
    for cell, value in cell_values.items():
        print(f"  {cell} = '{value}'")
    
    # Test Excel
    print("\n1️⃣ Testing Excel (requires Excel running with active workbook):")
    try:
        result = Agent.set_cell_values(cell_values, "Excel")
        print(f"Excel test result: {'✅ Success' if result else '❌ Failed'}")
    except Exception as e:
        print(f"Excel test exception: {e}")
    
    # Test LibreOffice Calc
    print("\n2️⃣ Testing LibreOffice Calc (requires LibreOffice running with UNO enabled):")
    
    # First check if UNO is available
    uno_available = False
    try:
        import uno
        from com.sun.star.connection import NoConnectException
        uno_available = True
        print("  📦 LibreOffice UNO bindings detected")
    except ImportError:
        print("  ❌ LibreOffice UNO bindings not available")
    except Exception as e:
        print(f"  ⚠️  Error checking UNO: {e}")
    
    # Now test the actual function
    try:
        result = Agent.set_cell_values(cell_values, "LibreOffice Calc")
        
        if not uno_available and result is True:
            print("  🚨 BUG: Function returned True without LibreOffice UNO!")
            print(f"LibreOffice test result: ❌ Bug Detected (returned True incorrectly)")
        elif not uno_available and result is False:
            print(f"LibreOffice test result: ✅ Correct (returned False as expected)")
        elif uno_available and result is True:
            print(f"LibreOffice test result: ✅ Success (UNO available and function succeeded)")
        elif uno_available and result is False:
            print(f"LibreOffice test result: ❌ Failed (UNO available but function failed)")
        else:
            print(f"LibreOffice test result: ⚠️  Unexpected return value: {result}")
            
    except Exception as e:
        print(f"LibreOffice test exception: {e}")
        print(f"LibreOffice test result: ❌ Failed (Exception occurred)")
    
    # Test via take_action
    print("\n3️⃣ Testing via take_action execution:")
    test_action = f"agent.set_cell_values({cell_values}, 'Excel')"
    try:
        result = take_action(test_action)
        print(f"take_action test result: {'✅ Success' if result else '❌ Failed'}")
    except Exception as e:
        print(f"take_action test exception: {e}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_set_cell_values()
