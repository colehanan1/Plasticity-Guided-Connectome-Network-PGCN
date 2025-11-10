#!/usr/bin/env python3
"""
Diagnose DoOR toolkit installation issues.

This script helps identify why DoOR toolkit imports might be failing.
"""

import sys
import importlib.util

print("="*80)
print("DoOR TOOLKIT INSTALLATION DIAGNOSTICS")
print("="*80)

# Test 1: Check if door_toolkit package exists
print("\n[1/5] Checking if door_toolkit package is installed...")
try:
    import door_toolkit
    print(f"✅ door_toolkit package found")
    print(f"   Location: {door_toolkit.__file__}")
    print(f"   Version: {getattr(door_toolkit, '__version__', 'unknown')}")
except ImportError as e:
    print(f"❌ door_toolkit package NOT found")
    print(f"   Error: {e}")
    print("\n   To install:")
    print("   cd ~/Documents/cole/VSCode/door-python-toolkit")
    print("   pip install -e .")
    sys.exit(1)

# Test 2: Check package structure
print("\n[2/5] Checking package structure...")
try:
    import os
    package_dir = os.path.dirname(door_toolkit.__file__)
    print(f"✅ Package directory: {package_dir}")

    # List contents
    print("\n   Contents:")
    for item in sorted(os.listdir(package_dir)):
        if not item.startswith('__pycache__'):
            path = os.path.join(package_dir, item)
            if os.path.isdir(path):
                print(f"   📁 {item}/")
            else:
                print(f"   📄 {item}")
except Exception as e:
    print(f"❌ Could not inspect package structure: {e}")

# Test 3: Check for integration module
print("\n[3/5] Checking for integration module...")
try:
    import door_toolkit.integration
    print(f"✅ door_toolkit.integration found")
    print(f"   Location: {door_toolkit.integration.__file__}")

    # List integration contents
    integration_dir = os.path.dirname(door_toolkit.integration.__file__)
    print("\n   Contents of integration/:")
    for item in sorted(os.listdir(integration_dir)):
        if not item.startswith('__pycache__') and item.endswith('.py'):
            print(f"   📄 {item}")
except ImportError as e:
    print(f"❌ door_toolkit.integration NOT found")
    print(f"   Error: {e}")
    print(f"\n   Expected structure:")
    print(f"   door_toolkit/")
    print(f"   └── integration/")
    print(f"       ├── __init__.py")
    print(f"       ├── encoder.py")
    print(f"       └── integrator.py")

# Test 4: Try importing DoOREncoder
print("\n[4/5] Trying to import DoOREncoder...")
try:
    from door_toolkit.integration.encoder import DoOREncoder
    print(f"✅ DoOREncoder imported successfully")
    print(f"   Class: {DoOREncoder}")
except ImportError as e:
    print(f"❌ Could not import DoOREncoder")
    print(f"   Error: {e}")

    # Try alternative import paths
    print(f"\n   Trying alternative import paths...")

    alternatives = [
        "door_toolkit.encoder.DoOREncoder",
        "door_toolkit.door_encoder.DoOREncoder",
        "door_toolkit.core.encoder.DoOREncoder",
    ]

    for alt_path in alternatives:
        module_path, class_name = alt_path.rsplit('.', 1)
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            print(f"   ✅ Found at: {alt_path}")
        except (ImportError, AttributeError):
            print(f"   ❌ Not at: {alt_path}")

# Test 5: Try importing DoORFlyWireIntegrator
print("\n[5/5] Trying to import DoORFlyWireIntegrator...")
try:
    from door_toolkit.integration.integrator import DoORFlyWireIntegrator
    print(f"✅ DoORFlyWireIntegrator imported successfully")
    print(f"   Class: {DoORFlyWireIntegrator}")
except ImportError as e:
    print(f"❌ Could not import DoORFlyWireIntegrator")
    print(f"   Error: {e}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

try:
    from door_toolkit.integration.encoder import DoOREncoder
    from door_toolkit.integration.integrator import DoORFlyWireIntegrator

    print("\n✅ DoOR toolkit is properly installed and importable!")
    print("\nYou can now use:")
    print("  - python scripts/test_door_integration.py")
    print("  - python scripts/test_or7a_veto.py (will use real DoOR data)")

    # Try creating an instance
    print("\n" + "="*80)
    print("TESTING FUNCTIONALITY")
    print("="*80)

    try:
        print("\nCreating DoOREncoder instance...")
        encoder = DoOREncoder()
        print("✅ DoOREncoder created successfully")

        print("\nGetting response matrix...")
        matrix = encoder.get_response_matrix()
        print(f"✅ Response matrix loaded: {matrix.shape[0]} odorants × {matrix.shape[1]} receptors")

        if 'Or7a' in matrix.columns and 'benzaldehyde' in matrix.index:
            response = matrix.loc['benzaldehyde', 'Or7a']
            print(f"✅ Sample data: Or7a response to benzaldehyde = {response:.3f}")
    except Exception as e:
        print(f"⚠️  Warning: Could not test functionality: {e}")

except ImportError:
    print("\n❌ DoOR toolkit is NOT properly installed")
    print("\nPlease check the error messages above and:")
    print("  1. Verify door-toolkit directory structure")
    print("  2. Re-install: cd ~/Documents/cole/VSCode/door-python-toolkit && pip install -e .")
    print("  3. Check for __init__.py files in integration/ directory")
    print("  4. Verify Python version compatibility")

print("\n" + "="*80)
