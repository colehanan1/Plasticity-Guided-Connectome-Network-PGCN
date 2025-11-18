#!/usr/bin/env python3
"""Quick script to find correct DoOREncoder method name."""

from door_toolkit.encoder import DoOREncoder

encoder = DoOREncoder()

print("DoOREncoder methods and attributes:")
print("="*60)

# Get all public methods/attributes
attrs = [attr for attr in dir(encoder) if not attr.startswith('_')]

for attr in sorted(attrs):
    obj = getattr(encoder, attr)
    if callable(obj):
        print(f"  {attr}() - method")
    else:
        print(f"  {attr} - attribute/property")
        try:
            value = obj
            if hasattr(value, 'shape'):
                print(f"      Shape: {value.shape}")
        except:
            pass

print("\n" + "="*60)
print("Testing likely matrix accessors:")

# Test common patterns
test_names = ['matrix', 'response_matrix', 'data', 'df', 'door_matrix']

for name in test_names:
    if hasattr(encoder, name):
        obj = getattr(encoder, name)
        if hasattr(obj, 'shape'):
            print(f"✅ {name}: {obj.shape}")
        elif callable(obj):
            try:
                result = obj()
                if hasattr(result, 'shape'):
                    print(f"✅ {name}(): {result.shape}")
            except:
                pass
