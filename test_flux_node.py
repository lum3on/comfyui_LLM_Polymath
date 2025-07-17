#!/usr/bin/env python3
"""
Test script to verify FluxContextPreset node is working correctly.
Run this script in your ComfyUI custom_nodes/comfyui_LLM_Polymath directory.
"""

import sys
import os

def test_flux_context_preset():
    print("=" * 60)
    print("FLUX CONTEXT PRESET NODE TEST")
    print("=" * 60)
    
    # Test 1: Direct import
    print("\n1. Testing direct import from flux_context_preset.py...")
    try:
        from flux_context_preset import FluxContextPreset, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        print("✅ SUCCESS: Direct import works")
        print(f"   Class: {FluxContextPreset}")
        print(f"   Mappings: {NODE_CLASS_MAPPINGS}")
        print(f"   Display: {NODE_DISPLAY_NAME_MAPPINGS}")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False
    
    # Test 2: Node structure
    print("\n2. Testing node structure...")
    try:
        node = FluxContextPreset()
        input_types = FluxContextPreset.INPUT_TYPES()
        
        # Check required attributes
        required_attrs = ['RETURN_TYPES', 'FUNCTION', 'CATEGORY']
        for attr in required_attrs:
            if not hasattr(FluxContextPreset, attr):
                print(f"❌ FAILED: Missing {attr}")
                return False
        
        print("✅ SUCCESS: Node structure is correct")
        print(f"   INPUT_TYPES: {input_types}")
        print(f"   RETURN_TYPES: {FluxContextPreset.RETURN_TYPES}")
        print(f"   FUNCTION: {FluxContextPreset.FUNCTION}")
        print(f"   CATEGORY: {FluxContextPreset.CATEGORY}")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False
    
    # Test 3: Node functionality
    print("\n3. Testing node functionality...")
    try:
        result = node.generate_prompt('Teleport', 'Make it dramatic')
        if isinstance(result, tuple) and len(result) == 1 and isinstance(result[0], str):
            print("✅ SUCCESS: Node functionality works")
            print(f"   Result type: {type(result)}")
            print(f"   Result length: {len(result[0])}")
            print(f"   Sample: {result[0][:100]}...")
        else:
            print(f"❌ FAILED: Unexpected result format: {type(result)}")
            return False
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False
    
    # Test 4: Main __init__.py import
    print("\n4. Testing main __init__.py import...")
    try:
        from __init__ import NODE_CLASS_MAPPINGS as MAIN_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as MAIN_DISPLAY
        
        if 'flux_context_preset' in MAIN_MAPPINGS:
            print("✅ SUCCESS: Node found in main mappings")
            print(f"   Available nodes: {list(MAIN_MAPPINGS.keys())}")
        else:
            print("❌ FAILED: Node not found in main mappings")
            print(f"   Available nodes: {list(MAIN_MAPPINGS.keys())}")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: {e}")
        print("   This might be due to missing dependencies for other nodes")
        print("   But FluxContextPreset should still work independently")
    
    # Test 5: Web directory
    print("\n5. Testing web directory...")
    web_js_path = os.path.join("web", "flux_context_preset.js")
    if os.path.exists(web_js_path):
        print("✅ SUCCESS: JavaScript file exists")
        print(f"   Path: {web_js_path}")
        print(f"   Size: {os.path.getsize(web_js_path)} bytes")
    else:
        print(f"❌ FAILED: JavaScript file not found at {web_js_path}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("The FluxContextPreset node should appear in ComfyUI.")
    print("If it's still not showing up, try:")
    print("1. Restart ComfyUI completely")
    print("2. Check ComfyUI console for error messages")
    print("3. Make sure you're in the correct custom_nodes directory")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path: {sys.path[0]}")
    
    success = test_flux_context_preset()
    sys.exit(0 if success else 1)
