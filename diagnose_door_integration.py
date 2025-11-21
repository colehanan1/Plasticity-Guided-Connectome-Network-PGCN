#!/usr/bin/env python3
"""
Diagnostic script for DoOR integration issues.

This script helps identify why DoOR odor → PN mapping is returning zero active PNs.
"""

import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

print("="*70)
print("DoOR Integration Diagnostic Script")
print("="*70)

# Step 1: Check which DoOR integration is being imported
print("\n[1/6] Checking DoOR integration imports...")
sys.path.insert(0, 'src')

try:
    from pgcn.data.door_integration import DoORIntegration
    print("✓ Successfully imported DoORIntegration from pgcn.data.door_integration")

    # Check if ODOR_NAME_MAP exists (from my fix)
    if hasattr(DoORIntegration, 'ODOR_NAME_MAP'):
        print(f"✓ ODOR_NAME_MAP found with {len(DoORIntegration.ODOR_NAME_MAP)} mappings")
        print("  Mappings:")
        for k, v in DoORIntegration.ODOR_NAME_MAP.items():
            print(f"    '{k}' → '{v}'")
    else:
        print("✗ ODOR_NAME_MAP NOT FOUND - you may not have pulled the latest fix!")
        print("  Run: git pull origin claude/connectome-constrained-behavior-prediction-014UV3FWTFdXYAttqMaTBEoh")
        sys.exit(1)

except ImportError as e:
    print(f"✗ Failed to import DoORIntegration: {e}")
    sys.exit(1)

# Step 2: Check cache directories
print("\n[2/6] Checking cache directories...")
cache_locations = [
    Path("data/cache"),
    Path("data/door_cache"),
    Path.home() / "Documents/cole/VSCode/Plasticity-Guided-Connectome-Network-PGCN-/data/cache",
    Path.home() / "Documents/cole/VSCode/Plasticity-Guided-Connectome-Network-PGCN-/data/door_cache",
]

cache_found = None
for cache_path in cache_locations:
    if cache_path.exists():
        print(f"  ✓ Found: {cache_path}")
        files = list(cache_path.glob("*"))
        print(f"    Files: {[f.name for f in files[:10]]}")
        cache_found = cache_path
    else:
        print(f"  ✗ Not found: {cache_path}")

if cache_found is None:
    print("⚠️  No cache directory found - DoOR will try to download from GitHub")
    cache_found = Path("data/cache")
    cache_found.mkdir(parents=True, exist_ok=True)
    print(f"  Created: {cache_found}")

# Step 3: Check for DoOR data files
print("\n[3/6] Checking for DoOR data files...")
door_files = [
    cache_found / "door_response_matrix.csv",
    cache_found / "response_matrix_norm.csv",
    cache_found / "response_matrix_norm.parquet",
    cache_found / "odorant_index.csv",
    cache_found / "odor_metadata.parquet",
]

door_data_found = False
for door_file in door_files:
    if door_file.exists():
        print(f"  ✓ Found: {door_file}")
        door_data_found = True
    else:
        print(f"  ✗ Not found: {door_file}")

if not door_data_found:
    print("⚠️  No DoOR data files found - will attempt to download")

# Step 4: Try initializing DoORIntegration
print("\n[4/6] Attempting to initialize DoORIntegration...")
try:
    door = DoORIntegration(cache_dir=cache_found)
    print(f"✓ DoORIntegration initialized successfully")
    print(f"  DoOR odorants loaded: {len(door.door_data)}")
    print(f"  DoOR columns (ORN types): {len(door.door_data.columns)}")
    print(f"  PNs mapped to glomeruli: {len(door.pn_glomeruli)}")

    # Show sample odor names from DoOR
    print(f"\n  Sample DoOR odor names (first 20):")
    for i, odor in enumerate(door.door_data.index[:20]):
        print(f"    {i+1:2d}. '{odor}'")

except FileNotFoundError as e:
    print(f"✗ FileNotFoundError: {e}")
    print("  Likely cause: nodes.parquet not found in cache directory")
    print("  Run FlyWire extraction scripts first")
    sys.exit(1)
except Exception as e:
    print(f"✗ Failed to initialize: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Test odor name resolution
print("\n[5/6] Testing odor name resolution...")
test_odors = [
    'hexanol',
    'ethyl_butyrate',
    'benzaldehyde',
    '3-octanol',
    'citral',
    'linalool',
    'apple_cider_vinegar',
]

resolution_results = {}
for odor in test_odors:
    resolved = door._resolve_odor_name(odor)
    resolution_results[odor] = resolved
    if resolved:
        print(f"  ✓ '{odor}' → '{resolved}'")
    else:
        print(f"  ✗ '{odor}' → NOT FOUND")

# Step 6: Test PN activity generation
print("\n[6/6] Testing PN activity generation...")
for odor in test_odors:
    try:
        pn_activity = door.odor_to_pn_activity(odor, n_pn=150)
        n_active = sum(pn_activity > 0.1)
        if n_active > 0:
            print(f"  ✓ {odor:25s}: {n_active:3d} active PNs")
        else:
            print(f"  ✗ {odor:25s}: {n_active:3d} active PNs (ZERO!)")
            # Debug: check if odor was resolved
            if resolution_results[odor]:
                print(f"      → Resolved to '{resolution_results[odor]}' but no PN activity")
                print(f"      → This suggests ORN→PN mapping issue")
            else:
                print(f"      → Odor name not resolved - name mapping issue")
    except Exception as e:
        print(f"  ✗ {odor:25s}: ERROR - {e}")

# Summary
print("\n" + "="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)

all_good = all(sum(door.odor_to_pn_activity(odor, n_pn=150) > 0.1) > 0
               for odor in test_odors)

if all_good:
    print("✅ All odors produce non-zero PN activity - DoOR integration is working!")
    print("\nIf train_ccbpn.py still shows 0 active PNs, check:")
    print("  1. Are you using the correct cache_dir path?")
    print("  2. Does the behavioral CSV have correct dataset names?")
    print("  3. Are odor names in the YAML exactly matching test_odors above?")
else:
    print("❌ Some odors produce zero PN activity - DoOR integration has issues!")
    print("\nPossible fixes:")
    print("  1. Pull latest code:")
    print("     git pull origin claude/connectome-constrained-behavior-prediction-014UV3FWTFdXYAttqMaTBEoh")
    print("  2. Check if nodes.parquet exists in cache directory")
    print("  3. Verify DoOR database is downloaded correctly")
    print("  4. Check PN→glomerulus mapping")

print("\nFor detailed logs, see output above.")
