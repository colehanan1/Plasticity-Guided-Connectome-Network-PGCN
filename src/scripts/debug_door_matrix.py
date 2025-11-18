#!/usr/bin/env python3
"""Debug DoOR matrix to find correct odorant/receptor names."""

from door_toolkit.encoder import DoOREncoder
import pandas as pd

print("="*80)
print("DEBUGGING DoOR MATRIX")
print("="*80)

encoder = DoOREncoder()

# Try different accessors
matrix = None
for attr_name in ['matrix', 'response_matrix', 'data', 'door_matrix', 'df']:
    if hasattr(encoder, attr_name):
        try:
            matrix = getattr(encoder, attr_name)
            if isinstance(matrix, pd.DataFrame) and len(matrix) > 0:
                print(f"\n✅ Found matrix at: encoder.{attr_name}")
                break
        except:
            continue

if matrix is None or not isinstance(matrix, pd.DataFrame):
    for method_name in ['get_matrix', 'get_response_matrix', 'get_data', 'to_dataframe']:
        if hasattr(encoder, method_name):
            try:
                method = getattr(encoder, method_name)
                if callable(method):
                    matrix = method()
                    if isinstance(matrix, pd.DataFrame) and len(matrix) > 0:
                        print(f"\n✅ Found matrix at: encoder.{method_name}()")
                        break
            except:
                continue

if matrix is None:
    print("❌ Could not find matrix!")
    exit(1)

print(f"\nMatrix shape: {matrix.shape}")
print(f"  Rows (index): {len(matrix.index)}")
print(f"  Columns: {len(matrix.columns)}")

print("\n" + "="*80)
print("CHECKING ORIENTATION")
print("="*80)

print(f"\nFirst 5 index names: {list(matrix.index[:5])}")
print(f"First 5 column names: {list(matrix.columns[:5])}")

# Check which orientation has receptors
receptors_to_find = ['Or7a', 'Or67b', 'Or22a', 'Or35a']
odorants_to_find = ['benzaldehyde', 'hexanol', '2-heptanone']

print("\n" + "="*80)
print("SEARCHING FOR RECEPTORS")
print("="*80)

receptors_in_index = [r for r in receptors_to_find if r in matrix.index]
receptors_in_columns = [r for r in receptors_to_find if r in matrix.columns]

print(f"\nReceptors in INDEX: {receptors_in_index}")
print(f"Receptors in COLUMNS: {receptors_in_columns}")

print("\n" + "="*80)
print("SEARCHING FOR ODORANTS")
print("="*80)

odorants_in_index = [o for o in odorants_to_find if o in matrix.index]
odorants_in_columns = [o for o in odorants_to_find if o in matrix.columns]

print(f"\nOdorants in INDEX: {odorants_in_index}")
print(f"Odorants in COLUMNS: {odorants_in_columns}")

# Search for similar names
print("\n" + "="*80)
print("FUZZY SEARCH - Index names containing 'Or7'")
print("="*80)
or7_in_index = [name for name in matrix.index if 'Or7' in str(name)]
print(f"Found {len(or7_in_index)}: {or7_in_index[:10]}")

print("\n" + "="*80)
print("FUZZY SEARCH - Column names containing 'Or7'")
print("="*80)
or7_in_columns = [name for name in matrix.columns if 'Or7' in str(name)]
print(f"Found {len(or7_in_columns)}: {or7_in_columns[:10]}")

print("\n" + "="*80)
print("FUZZY SEARCH - Index names containing 'benz'")
print("="*80)
benz_in_index = [name for name in matrix.index if 'benz' in str(name).lower()]
print(f"Found {len(benz_in_index)}: {benz_in_index}")

print("\n" + "="*80)
print("FUZZY SEARCH - Column names containing 'benz'")
print("="*80)
benz_in_columns = [name for name in matrix.columns if 'benz' in str(name).lower()]
print(f"Found {len(benz_in_columns)}: {benz_in_columns}")

# Sample some actual values
print("\n" + "="*80)
print("SAMPLE DATA")
print("="*80)
print(f"\nFirst 5x5 cells:")
print(matrix.iloc[:5, :5])

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

if len(receptors_in_columns) > 0 and len(odorants_in_index) > 0:
    print("\n✅ Matrix is: odorants (rows) × receptors (columns)")
    print(f"   Access: matrix.loc['benzaldehyde', 'Or7a']")
elif len(receptors_in_index) > 0 and len(odorants_in_columns) > 0:
    print("\n✅ Matrix is: receptors (rows) × odorants (columns)")
    print(f"   Access: matrix.loc['Or7a', 'benzaldehyde']")
else:
    print("\n⚠️  Could not find standard names. Check fuzzy search results above.")
    print(f"   You may need to use different odorant/receptor names")

print("\n" + "="*80)
