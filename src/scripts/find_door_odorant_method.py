#!/usr/bin/env python3
"""Find how to access odorants by name in DoOREncoder."""

from door_toolkit.encoder import DoOREncoder
import pandas as pd

encoder = DoOREncoder()

print("="*80)
print("DoOREncoder Methods for Odorant Lookup")
print("="*80)

# Get all methods
methods = [m for m in dir(encoder) if not m.startswith('_') and callable(getattr(encoder, m))]

print(f"\nAll public methods:")
for method in sorted(methods):
    print(f"  - {method}()")

# Check for odorant-related methods
print("\n" + "="*80)
print("Methods containing 'odorant' or 'odor':")
print("="*80)
odorant_methods = [m for m in methods if 'odor' in m.lower()]
for method in odorant_methods:
    print(f"  - {method}()")

# Check for name/lookup methods
print("\n" + "="*80)
print("Methods containing 'name', 'lookup', 'get', or 'find':")
print("="*80)
lookup_methods = [m for m in methods if any(word in m.lower() for word in ['name', 'lookup', 'get', 'find', 'search'])]
for method in lookup_methods:
    print(f"  - {method}()")

# Check attributes
print("\n" + "="*80)
print("Checking for odorant name mappings:")
print("="*80)

attrs_to_check = ['odorant_names', 'odorants', 'odor_names', 'name_map', 'inchikey_to_name', 'name_to_inchikey']
for attr in attrs_to_check:
    if hasattr(encoder, attr):
        val = getattr(encoder, attr)
        print(f"\n✅ Found: encoder.{attr}")
        if isinstance(val, dict):
            print(f"   Type: dict with {len(val)} entries")
            # Show a few examples
            for i, (k, v) in enumerate(list(val.items())[:3]):
                print(f"   Example: {k} -> {v}")
        elif isinstance(val, (list, pd.Series, pd.Index)):
            print(f"   Type: {type(val).__name__} with {len(val)} entries")
            print(f"   First 5: {list(val)[:5]}")

# Try to get response for a specific odorant by name
print("\n" + "="*80)
print("Testing response lookup methods:")
print("="*80)

test_methods = ['get_response', 'get_odorant_response', 'get_odor_response', 'response']
test_odorant = 'benzaldehyde'
test_receptor = 'Or7a'

for method_name in test_methods:
    if hasattr(encoder, method_name):
        try:
            method = getattr(encoder, method_name)
            print(f"\n✅ Found method: encoder.{method_name}")
            # Try different argument patterns
            try:
                result = method(test_odorant, test_receptor)
                print(f"   {method_name}('{test_odorant}', '{test_receptor}') = {result}")
            except Exception as e:
                print(f"   Call with 2 args failed: {e}")

            try:
                result = method(test_odorant)
                print(f"   {method_name}('{test_odorant}') = {result}")
            except Exception as e:
                print(f"   Call with 1 arg failed: {e}")
        except:
            pass

print("\n" + "="*80)
