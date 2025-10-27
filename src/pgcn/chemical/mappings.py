"""Static chemical annotations derived from behavioral experiments."""

from __future__ import annotations

from typing import Mapping, Tuple

# Odor mappings keyed by training condition and testing slot.
COMPLETE_ODOR_MAPPINGS: Mapping[str, Mapping[str, str]] = {
    "opto_EB": {
        "testing_1": "hexanol",
        "testing_2": "ethyl_butyrate",
        "testing_3": "hexanol",
        "testing_4": "ethyl_butyrate",
        "testing_5": "ethyl_butyrate",
        "testing_6": "apple_cider_vinegar",
        "testing_7": "3-octanol",
        "testing_8": "benzaldehyde",
        "testing_9": "citral",
        "testing_10": "linalool",
    },
    "opto_benz_1": {
        "testing_1": "hexanol",
        "testing_2": "benzaldehyde",
        "testing_3": "hexanol",
        "testing_4": "benzaldehyde",
        "testing_5": "benzaldehyde",
        "testing_6": "apple_cider_vinegar",
        "testing_7": "3-octanol",
        "testing_8": "ethyl_butyrate",
        "testing_9": "citral",
        "testing_10": "linalool",
    },
    "opto_hex": {
        "testing_1": "apple_cider_vinegar",
        "testing_2": "hexanol",
        "testing_3": "apple_cider_vinegar",
        "testing_4": "hexanol",
        "testing_5": "hexanol",
        "testing_6": "benzaldehyde",
        "testing_7": "3-octanol",
        "testing_8": "ethyl_butyrate",
        "testing_9": "citral",
        "testing_10": "linalool",
    },
    "hex_control": {
        "testing_1": "apple_cider_vinegar",
        "testing_2": "hexanol",
        "testing_3": "apple_cider_vinegar",
        "testing_4": "hexanol",
        "testing_5": "hexanol",
        "testing_6": "benzaldehyde",
        "testing_7": "3-octanol",
        "testing_8": "ethyl_butyrate",
        "testing_9": "citral",
        "testing_10": "linalool",
    },
}

# Physicochemical descriptors distilled from literature and vendor data.
CHEMICAL_PROPERTIES: Mapping[str, Mapping[str, object]] = {
    "ethyl_butyrate": {
        "class": "ester",
        "molecular_weight": 116.16,
        "functional_groups": ["ester"],
        "carbon_length": 6,
        "boiling_point": 121.0,
        "odor_descriptor": ["fruity", "sweet"],
    },
    "benzaldehyde": {
        "class": "aldehyde",
        "molecular_weight": 106.12,
        "functional_groups": ["aldehyde", "aromatic"],
        "carbon_length": 7,
        "boiling_point": 179.0,
        "odor_descriptor": ["almond", "sweet"],
    },
    "hexanol": {
        "class": "alcohol",
        "molecular_weight": 102.17,
        "functional_groups": ["alcohol"],
        "carbon_length": 6,
        "boiling_point": 157.0,
        "odor_descriptor": ["floral", "green"],
    },
    "3-octanol": {
        "class": "alcohol",
        "molecular_weight": 130.23,
        "functional_groups": ["alcohol"],
        "carbon_length": 8,
        "boiling_point": 176.0,
        "odor_descriptor": ["fatty", "mushroom"],
    },
    "apple_cider_vinegar": {
        "class": "acid",
        "molecular_weight": 60.05,
        "functional_groups": ["acid"],
        "carbon_length": 2,
        "boiling_point": 118.0,
        "odor_descriptor": ["sour", "pungent"],
    },
    "citral": {
        "class": "terpene_aldehyde",
        "molecular_weight": 152.23,
        "functional_groups": ["aldehyde", "terpene"],
        "carbon_length": 10,
        "boiling_point": 229.0,
        "odor_descriptor": ["citrus", "lemon"],
    },
    "linalool": {
        "class": "terpene_alcohol",
        "molecular_weight": 154.25,
        "functional_groups": ["alcohol", "terpene"],
        "carbon_length": 10,
        "boiling_point": 198.0,
        "odor_descriptor": ["floral", "lavender"],
    },
}

# Precomputed similarity scores blending structural and functional motifs.
CHEMICAL_SIMILARITY_MATRIX: Mapping[Tuple[str, str], float] = {
    ("ethyl_butyrate", "hexanol"): 0.3,
    ("ethyl_butyrate", "3-octanol"): 0.4,
    ("ethyl_butyrate", "benzaldehyde"): 0.2,
    ("hexanol", "3-octanol"): 0.8,
    ("hexanol", "linalool"): 0.5,
    ("benzaldehyde", "citral"): 0.6,
    ("citral", "linalool"): 0.7,
    ("apple_cider_vinegar", "*"): 0.1,
}

# Behavioral priors derived from optogenetic conditioning readouts.
EMPIRICAL_RESPONSE_PATTERNS: Mapping[str, Mapping[object, object]] = {
    "training_success": {
        "ethyl_butyrate": 0.80,
        "hexanol": 0.84,
        "benzaldehyde": 0.26,
    },
    "cross_odor_generalization": {
        ("ethyl_butyrate", "hexanol"): 0.61,
        ("ethyl_butyrate", "3-octanol"): 0.50,
        ("hexanol", "3-octanol"): 0.50,
        ("all_odors", "apple_cider_vinegar"): 0.045,
    },
    "innate_valence": {
        "apple_cider_vinegar": -0.8,
        "linalool": -0.1,
        "3-octanol": 0.0,
        "citral": -0.2,
    },
}

# Simplified odor→glomerulus activation templates curated from the literature.
ODOR_GLOMERULUS_MAPPING: Mapping[str, Mapping[str, float]] = {
    "ethyl_butyrate": {"DM1": 0.8, "DL3": 0.6, "VM2": 0.3},
    "benzaldehyde": {"DM2": 0.9, "VA1d": 0.7, "DL1": 0.4},
    "hexanol": {"DL1": 0.8, "VM2": 0.6, "VA1d": 0.2},
    "3-octanol": {"DL1": 0.6, "VM2": 0.5, "DM1": 0.3},
    "apple_cider_vinegar": {"DM2": 0.9, "VM1": 0.8, "DL3": 0.4},
}

__all__ = [
    "COMPLETE_ODOR_MAPPINGS",
    "CHEMICAL_PROPERTIES",
    "CHEMICAL_SIMILARITY_MATRIX",
    "EMPIRICAL_RESPONSE_PATTERNS",
    "ODOR_GLOMERULUS_MAPPING",
]
