#!/usr/bin/env python3
"""
Diagnose DoOR toolkit import issues

This script systematically tests different import methods to identify
why the door-python-toolkit package isn't being detected.
"""

import sys
import subprocess
import os

print("=" * 80)
print("DOOR TOOLKIT IMPORT DIAGNOSTICS")
print("=" * 80)

# Check Python version and environment
print(f"\n1. Python Version: {sys.version}")
print(f"   Executable: {sys.executable}")

# Check sys.path
print("\n2. sys.path (first 10 entries):")
for i, path in enumerate(sys.path[:10]):
    print(f"   [{i}] {path}")

# Try different import variations
print("\n3. Testing import variations:")

import_attempts = [
    ("import door", None),
    ("import door_toolkit", None),
    ("import door_python_toolkit", None),
    ("from door import DoOREncoder", "DoOREncoder"),
    ("from door.encoder import DoOREncoder", "DoOREncoder"),
    ("from door_toolkit import DoOREncoder", "DoOREncoder"),
    ("from door_toolkit.encoder import DoOREncoder", "DoOREncoder"),
]

successful_import = None
successful_module = None

for attempt, class_name in import_attempts:
    try:
        exec(attempt)
        print(f"   ✅ {attempt}")

        # If successful, show module info
        if '.' in attempt.split()[1]:
            module_name = attempt.split()[1].split('.')[0]
        else:
            module_name = attempt.split()[1]

        module = sys.modules.get(module_name)
        if module:
            print(f"      Location: {getattr(module, '__file__', 'built-in')}")
            print(f"      Version: {getattr(module, '__version__', 'unknown')}")

            # Show available attributes
            attrs = [x for x in dir(module) if not x.startswith('_')]
            print(f"      Attributes ({len(attrs)}): {attrs[:10]}...")

            successful_import = attempt
            successful_module = module_name

            # If we got DoOREncoder, show its methods
            if class_name:
                try:
                    encoder_class = eval(class_name)
                    print(f"      {class_name} methods: {[m for m in dir(encoder_class) if not m.startswith('_')][:8]}...")
                except:
                    pass
        break

    except ImportError as e:
        print(f"   ❌ {attempt}")
        print(f"      ImportError: {e}")
    except Exception as e:
        print(f"   ⚠️  {attempt} - Other error: {type(e).__name__}: {e}")

# Check if package is installed
print("\n4. Checking pip installation:")
try:
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'show', 'door-python-toolkit'],
        capture_output=True,
        text=True,
        timeout=10
    )
    if result.stdout:
        print(result.stdout)
    else:
        print("   ⚠️  Package not found in pip list")

        # Try alternate package names
        for pkg_name in ['door', 'door-toolkit', 'doorpy']:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'show', pkg_name],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.stdout:
                print(f"\n   Found alternate package: {pkg_name}")
                print(result.stdout)
                break

except Exception as e:
    print(f"   ❌ Failed to check pip: {e}")

# Try importing with explicit path
print("\n5. Attempting direct site-packages import:")

# Find site-packages directories
site_packages_candidates = [
    p for p in sys.path
    if 'site-packages' in p or 'dist-packages' in p
]

print(f"   Found {len(site_packages_candidates)} site-packages directories:")
for sp in site_packages_candidates[:3]:
    print(f"   - {sp}")

    # Check if door exists there
    door_candidates = [
        os.path.join(sp, 'door'),
        os.path.join(sp, 'door_toolkit'),
        os.path.join(sp, 'door_python_toolkit'),
    ]

    for candidate in door_candidates:
        if os.path.exists(candidate):
            print(f"     ✅ Found: {candidate}")

            # List contents
            try:
                contents = os.listdir(candidate)
                py_files = [f for f in contents if f.endswith('.py')]
                print(f"        Python files: {py_files[:5]}...")
            except:
                pass

# Try to load DoOREncoder and test it
print("\n6. Testing DoOREncoder functionality:")
if successful_import:
    print(f"   Using successful import: {successful_import}")

    try:
        exec(successful_import)

        # Try to create an encoder instance
        if 'DoOREncoder' in dir():
            encoder = DoOREncoder()
            print(f"   ✅ DoOREncoder instantiated successfully")

            # Check available methods
            methods = [m for m in dir(encoder) if not m.startswith('_')]
            print(f"   Available methods: {methods}")

            # Try to access data
            for attr in ['matrix', 'response_matrix', 'data', 'door_matrix', 'df', 'odorant_names', 'receptor_names']:
                if hasattr(encoder, attr):
                    val = getattr(encoder, attr)
                    print(f"   ✅ encoder.{attr}: {type(val).__name__}", end="")

                    # If it's a DataFrame or similar, show shape
                    if hasattr(val, 'shape'):
                        print(f" shape={val.shape}")
                    elif isinstance(val, (list, tuple)):
                        print(f" length={len(val)}")
                    else:
                        print()

            # Try encode method
            if hasattr(encoder, 'encode'):
                print("\n   Testing encode() method:")
                try:
                    result = encoder.encode('benzaldehyde')
                    print(f"   ✅ encoder.encode('benzaldehyde'): {type(result).__name__}")
                    if hasattr(result, 'shape'):
                        print(f"      Shape: {result.shape}")
                    if hasattr(result, '__len__'):
                        print(f"      First 5 values: {result[:5]}")
                except Exception as e:
                    print(f"   ❌ encode() failed: {e}")

        else:
            print("   ⚠️  DoOREncoder not available after import")

    except Exception as e:
        print(f"   ❌ Failed to test DoOREncoder: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
else:
    print("   ⚠️  No successful import found - cannot test DoOREncoder")

# Final recommendation
print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

if successful_import:
    print(f"✅ Use this import statement in your scripts:")
    print(f"   {successful_import}")

    if successful_module:
        module = sys.modules.get(successful_module)
        if module and hasattr(module, '__file__'):
            print(f"\n   Module location: {module.__file__}")
else:
    print("❌ DoOR toolkit is NOT properly installed or importable")
    print("\nTroubleshooting steps:")
    print("1. Install/reinstall the package:")
    print("   pip install --upgrade door-python-toolkit")
    print("\n2. Or try installing from GitHub:")
    print("   pip install git+https://github.com/your-repo/door-python-toolkit.git")
    print("\n3. Check if conda environment is active:")
    print(f"   Current Python: {sys.executable}")
    print("   Expected: /home/ramanlab/anaconda3/envs/PGCN/bin/python")

print("=" * 80)
