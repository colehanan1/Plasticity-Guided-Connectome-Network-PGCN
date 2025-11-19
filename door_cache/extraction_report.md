# DoOR Data Extraction Report

**Generated:** 2025-11-05 20:25:06  
**Source:** DoOR v2.0.0 (Database of Odorant Responses)

## Dataset Overview

- **Odorants:** 693 unique odorant compounds
- **Receptors:** 78 ORN/receptor types
- **Total measurements:** 7,381
- **Sparsity:** 86.35% missing values
- **Value range:** [0.00, 1.00]

## Top Receptors by Coverage

| Receptor | Odorants Tested | Coverage % |
|----------|----------------|------------|
| Or19a | 497 | 71.7% |
| Or10a | 235 | 33.9% |
| Or22a | 225 | 32.5% |
| Or7a | 222 | 32.0% |
| ac4 | 190 | 27.4% |
| ab4B | 182 | 26.3% |
| Or82a | 180 | 26.0% |
| Or47b | 178 | 25.7% |
| Or42b | 177 | 25.5% |
| Or92a | 174 | 25.1% |

## Glomerular Distribution

['ac1A', 'ac1B', 'ac2A', 'ac2B', 'ac3A', 'ac3B', 'ac4', 'Or1a', 'Or2a', 'Or7a', 'Or9a', 'Or10a', 'Or13a', 'Or19a', 'Or22a', 'Or22b', 'Or22c', 'Or23a', 'Or24a', 'Or30a', 'Or33a', 'Or33b', 'Or33c', 'Or35a', 'Or42a', 'Or42b', 'Or43a', 'Or43b', 'Or45a', 'Or45b', 'Or46a', 'Or47a', 'Or47b', 'Or49a', 'Or49b', 'Or59a', 'Or59b', 'Or59c', 'Or65a', 'Or67a', 'Or67b', 'Or67c', 'Or67d', 'Or71a', 'Or74a', 'Or82a', 'Or85a', 'Or85b', 'Or85c', 'Or85d', 'Or85e', 'Or85f', 'Or88a', 'Or92a', 'Or94a', 'Or94b', 'Or98a', 'Gr21a.Gr63a', 'ab2B', 'ab4B', 'ab5B', 'pb2A', 'Or69a', 'ac1', 'ac2', 'ac3_noOr35a', 'Ir31a', 'Ir41a', 'Ir75a', 'Ir75d', 'Ir76a', 'Ir84a', 'Ir92a', 'Ir64a.DC4', 'Ir64a.DP1m', 'ac1BC', 'ac2BC', 'Or83c']

## Usage in PGCN Project

### Load Response Matrix (Recommended)

```python
import pandas as pd
import numpy as np

# Load normalized response matrix (odorants × receptors)
response_df = pd.read_parquet('data/door_cache/response_matrix_norm.parquet')

# Or load as numpy array
response_array = np.load('data/door_cache/response_matrix_norm.npy')
receptor_names = pd.read_csv('data/door_cache/receptor_index.csv')['receptor'].tolist()
```

### Create Odor-PN Encoder

```python
# Map odorants to PN (glomerular) activation patterns
def door_odor_to_pn(odor_name: str, response_df: pd.DataFrame) -> np.ndarray:
    '''Convert odor name to PN activation vector.'''
    if odor_name not in response_df.index:
        raise KeyError(f"Odor {odor_name} not in DoOR database")
    
    pn_response = response_df.loc[odor_name].fillna(0.0).values
    return pn_response

# Example: Get ethyl acetate response
ethyl_acetate_pn = door_odor_to_pn('ethyl acetate', response_df)
```

### Integration with PGCN

```python
# In pgcn/encoders.py
class DoOREncoder:
    def __init__(self, door_cache_path: str):
        self.response_matrix = pd.read_parquet(f'{door_cache_path}/response_matrix_norm.parquet')
        self.n_channels = self.response_matrix.shape[1]
    
    def encode(self, odor_name: str) -> torch.Tensor:
        pn_activation = self.response_matrix.loc[odor_name].fillna(0.0).values
        return torch.from_numpy(pn_activation).float()
```

## Files Generated

- `response_matrix_norm.parquet` - Normalized response matrix (recommended)
- `response_matrix_norm.csv` - CSV format for inspection
- `response_matrix_norm.npy` - NumPy array for training
- `odor_metadata.parquet` - Odorant chemical properties
- `al_map.parquet` - Receptor-to-glomerulus mapping
- `receptor_index.csv` - Ordered receptor names
- `odorant_index.csv` - Ordered odorant names
- `metadata.json` - Extraction metadata & data hash

## Next Steps

1. Verify receptor names match your connectome annotation (Or10a, Or22a, etc.)
2. Decide on PN channel count (use all 78 or subset)
3. Handle missing values (fill with 0, mean, or train a predictor)
4. Integrate into `pgcn/encoders.py` as `DoOREncoder` class

---
*DoOR Database: Münch & Galizia (2016), Scientific Data 3:160122*
