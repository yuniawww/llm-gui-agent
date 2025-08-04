#!/usr/bin/env python3
"""
Test complete grounding workflow: Runner → Grounding → LLM → Execution
Simulates the full pipeline from runner output to action execution
"""

import sys
import os
import time

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grounding import AccessibilityGrounding, ground_runner_action, take_action


def test_complete_grounding_workflow():
    """Test the complete grounding workflow"""
    print("=== Complete Grounding Workflow Test ===")
    print("Testing: Runner → Grounding → LLM → Execution")
    print()
    
    # Step 1: Mock runner results (typical runner output)
    print("📋 Step 1: Mock Runner Results")
    print("-" * 40)
    
    test_cases = [
        {
            "name": "Click on File menu",
            "runner_result": {
                "screen_analysis": "I can see VS Code window with a menu bar at the top. The File menu is visible in the top-left area.",
                "next_action": "Click on the File menu to open it",
                "grounded_action": "agent.click('File menu in menu bar', 1, 'left', [])"
            }
        },
        {
            "name": "Click on Edit menu", 
            "runner_result": {
                "screen_analysis": "VS Code interface with menu bar containing File, Edit, Selection, View options.",
                "next_action": "Click on Edit menu to access editing options",
                "grounded_action": "agent.click('Edit menu item', 1, 'left', [])"
            }
        },
        {
            "name": "Toggle side panel",
            "runner_result": {
                "screen_analysis": "VS Code with side panel toggle button visible in the top-right corner.",
                "next_action": "Click the toggle side panel button to show/hide the side panel",
                "grounded_action": "agent.click('Toggle Primary Side Bar button', 1, 'left', [])"
            }
        },
        {
            "name": "Safe completion action",
            "runner_result": {
                "screen_analysis": "Task completed successfully",
                "next_action": "Mark task as done",
                "grounded_action": "agent.done()"
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"🧪 Test Case {i}: {test_case['name']}")
        print(f"{'='*60}")
        
        runner_result = test_case["runner_result"]
        
        # Display runner output
        print("📥 Runner Output:")
        print(f"  Screen Analysis: {runner_result['screen_analysis']}")
        print(f"  Next Action: {runner_result['next_action']}")
        print(f"  Grounded Action: {runner_result['grounded_action']}")
        
        # Step 2: Grounding Process
        print("\n🔍 Step 2: Grounding Process")
        print("-" * 40)
        
        try:
            grounder = AccessibilityGrounding()
            
            # Get window info
            handle, title = grounder.tree_extractor.get_foreground_window_info()
            print(f"Target Window: {title}")
            print(f"Window Handle: {handle}")
            
            if not handle:
                print("❌ No window available for grounding")
                continue
            
            # Get accessibility tree
            tree_data = grounder.tree_extractor.get_window_accessibility_tree(handle)
            if tree_data:
                print(f"✅ Accessibility tree extracted ({len(tree_data.get('children', []))} top-level children)")
            else:
                print("❌ Failed to extract accessibility tree")
                continue
            
            # Step 3: LLM Grounding
            print("\n🤖 Step 3: LLM Grounding")
            print("-" * 40)
            print("Calling LLM to convert descriptions to coordinates...")
            
            # This calls the LLM to convert the runner's grounded_action to actual coordinates
            grounded_code = grounder.ground_action(runner_result)
            
            print(f"📤 LLM Input: {runner_result['grounded_action']}")
            print(f"📥 LLM Output: {grounded_code}")
            
            # Step 4: Action Execution (Simulated)
            print("\n⚡ Step 4: Action Execution")
            print("-" * 40)
            
            if grounded_code == "agent.done()":
                print("🎯 Safe action detected - executing immediately")
                result = take_action(grounded_code)
                print(f"✅ Execution result: {result}")
            elif grounded_code == "agent.fail()":
                print("❌ Grounding failed - no action taken")
            elif grounded_code.startswith("agent.click(") and "," in grounded_code:
                # Extract coordinates for verification
                try:
                    # Parse coordinates from the grounded code
                    import re
                    coord_match = re.search(r'agent\.click\((\d+),\s*(\d+)', grounded_code)
                    if coord_match:
                        x, y = int(coord_match.group(1)), int(coord_match.group(2))
                        print(f"🎯 Target coordinates: ({x}, {y})")
                        print(f"📋 Full action: {grounded_code}")
                        print("⚠️  Would execute click (simulated for safety)")
                        # result = take_action(grounded_code)  # Uncomment to actually execute
                        result = True  # Simulated success
                        print(f"✅ Simulated execution result: {result}")
                    else:
                        print(f"⚠️  Could not parse coordinates from: {grounded_code}")
                except Exception as e:
                    print(f"❌ Error parsing action: {e}")
            else:
                print(f"⚠️  Unknown action format: {grounded_code}")
            
            # Step 5: Results Summary
            print("\n📊 Step 5: Results Summary")
            print("-" * 40)
            print(f"✅ Runner → Grounding: SUCCESS")
            print(f"✅ Grounding → LLM: SUCCESS") 
            print(f"✅ LLM → Coordinates: {'SUCCESS' if grounded_code != 'agent.fail()' else 'FAILED'}")
            print(f"✅ Workflow Status: {'COMPLETE' if grounded_code != 'agent.fail()' else 'FAILED'}")
            
        except Exception as e:
            print(f"❌ Error in test case {i}: {e}")
            import traceback
            traceback.print_exc()
        
        # Wait between test cases
        if i < len(test_cases):
            print(f"\n⏳ Waiting 2 seconds before next test case...")
            time.sleep(2)
    
    print(f"\n{'='*60}")
    print("🏁 Complete Workflow Test Finished")
    print(f"{'='*60}")


def test_coordinate_accuracy():
    """Test the accuracy of coordinate generation"""
    print("\n=== Coordinate Accuracy Test ===")
    
    try:
        grounder = AccessibilityGrounding()
        
        # Get current accessibility tree for reference
        handle, title = grounder.tree_extractor.get_foreground_window_info()
        if not handle:
            print("No window available for coordinate testing")
            return
        
        tree_data = grounder.tree_extractor.get_window_accessibility_tree(handle)
        if not tree_data:
            print("No accessibility tree available")
            return
        
        # Extract known elements with coordinates
        formatter = grounder.tree_formatter
        clickable_elements = formatter.extract_clickable_elements(tree_data)
        
        print(f"Found {len(clickable_elements)} clickable elements")
        print("\n📍 Known Element Coordinates:")
        
        for i, elem in enumerate(clickable_elements[:5]):  # Test first 5 elements
            print(f"{i+1}. {elem['name'][:30]:30} at ({elem['center_x']:4d}, {elem['center_y']:4d})")
        
        # Test grounding for a specific known element
        if clickable_elements:
            test_element = clickable_elements[0]  # Use first element
            
            mock_runner_result = {
                "screen_analysis": f"I can see a {test_element['type']} element named '{test_element['name']}'",
                "next_action": f"Click on the {test_element['name']} element",
                "grounded_action": f"agent.click('{test_element['name']}', 1, 'left', [])"
            }
            
            print(f"\n🎯 Testing coordinate accuracy for: {test_element['name']}")
            print(f"Expected coordinates: ({test_element['center_x']}, {test_element['center_y']})")
            
            grounded_code = grounder.ground_action(mock_runner_result)
            print(f"LLM generated: {grounded_code}")
            
            # Compare with expected coordinates
            import re
            coord_match = re.search(r'agent\.click\((\d+),\s*(\d+)', grounded_code)
            if coord_match:
                generated_x, generated_y = int(coord_match.group(1)), int(coord_match.group(2))
                error_x = abs(generated_x - test_element['center_x'])
                error_y = abs(generated_y - test_element['center_y'])
                
                print(f"Generated coordinates: ({generated_x}, {generated_y})")
                print(f"Coordinate error: ({error_x}, {error_y}) pixels")
                
                if error_x <= 10 and error_y <= 10:
                    print("✅ Coordinate accuracy: EXCELLENT (within 10 pixels)")
                elif error_x <= 50 and error_y <= 50:
                    print("✅ Coordinate accuracy: GOOD (within 50 pixels)")
                else:
                    print("⚠️  Coordinate accuracy: NEEDS IMPROVEMENT (>50 pixels error)")
            else:
                print("❌ Could not extract coordinates from LLM output")
        
    except Exception as e:
        print(f"Error in coordinate accuracy test: {e}")


if __name__ == "__main__":
    print("🚀 Testing Complete Grounding Workflow")
    print("=" * 60)
    print("This test simulates the full pipeline:")
    print("1. Runner generates grounded_action with descriptions")
    print("2. Grounding extracts accessibility tree") 
    print("3. LLM converts descriptions to coordinates")
    print("4. Action executor runs the final code")
    print("=" * 60)
    
    test_complete_grounding_workflow()
    test_coordinate_accuracy()
    
    print("\n✨ All tests completed!")
    print("Note: Actual clicking is simulated for safety.")
    print("Remove simulation to enable real interaction.")
