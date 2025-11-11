#!/usr/bin/env python3
"""Test DoOR odorant name lookup."""

from door_toolkit.encoder import DoOREncoder

encoder = DoOREncoder()

print("="*80)
print("Testing Odorant Name Lookup")
print("="*80)

# Check if benzaldehyde is in the list
test_odorants = ['benzaldehyde', 'hexanol', '2-heptanone', 'Benzaldehyde', 'BENZALDEHYDE']

print("\nSearching for odorant names:")
for odorant in test_odorants:
    if odorant in encoder.odorant_names:
        print(f"  ✅ Found: '{odorant}'")
        inchikey = encoder.name_to_inchikey.get(odorant)
        print(f"     InChIKey: {inchikey}")
    else:
        print(f"  ❌ Not found: '{odorant}'")

# Fuzzy search
print("\n" + "="*80)
print("Fuzzy search in odorant_names:")
print("="*80)

for search_term in ['benz', 'hexan', 'heptan']:
    matches = [name for name in encoder.odorant_names if search_term in name.lower()]
    print(f"\nContains '{search_term}': {len(matches)} matches")
    for match in matches[:5]:
        print(f"  - {match}")

# Try the encode method
print("\n" + "="*80)
print("Testing encode() method:")
print("="*80)

try:
    # Try different odorant name formats
    for odorant in ['benzaldehyde', 'Benzaldehyde', 'sfr']:
        try:
            result = encoder.encode(odorant)
            print(f"\n✅ encode('{odorant}'):")
            print(f"   Type: {type(result)}")
            print(f"   Shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
            if hasattr(result, 'index'):
                print(f"   Index: {list(result.index)[:5]}")
            print(f"   Sample values: {result.values[:5] if hasattr(result, 'values') else result}")
        except Exception as e:
            print(f"\n❌ encode('{odorant}') failed: {e}")
except Exception as e:
    print(f"encode() method not working as expected: {e}")

print("\n" + "="*80)
