"""
Comprehensive Cell Type Summary for PGCN Olfactory System Model

This script generates a complete inventory of all neuron types integrated into
the PGCN model, including counts, connectivity statistics, and functional roles.

Usage:
    python scripts/summarize_all_cell_types.py [--cache-dir data/cache]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loaders.circuit_loader import CircuitLoader


def generate_cell_type_summary(cache_dir: Path) -> Dict[str, Any]:
    """
    Generate comprehensive summary of all cell types in PGCN model.

    Parameters
    ----------
    cache_dir : Path
        Path to cache directory containing extracted neuron CSVs

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - total_neurons: int
        - total_connections: int
        - cell_type_breakdown: List[dict] with (name, count, role, neurotransmitter)
        - connectivity_summary: Matrix dimensions for each pathway
    """
    # Load CSVs directly from cache
    cell_types: List[Dict[str, Any]] = []

    # Helper to load CSV and count neurons
    def load_csv_count(filename: str) -> int:
        csv_path = cache_dir / filename
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            return len(df)
        return 0

    # Core Components
    pn_count = load_csv_count("alpn_extracted.csv")
    cell_types.append({
        "name": "Projection Neurons (PNs)",
        "count": pn_count,
        "role": "Odor-responsive sensory input from antennal lobe",
        "neurotransmitter": "Cholinergic/Glutamatergic",
        "connectivity": "PN→KC (sparse expansion coding)",
        "category": "Core",
    })

    # KC subtypes
    kc_subtypes = [
        ("kc_ab.csv", "α/β KCs (long-term memory)"),
        ("kc_ab_p.csv", "α/β posterior KCs"),
        ("kc_apbp_main.csv", "α'/β' main KCs (intermediate memory)"),
        ("kc_apbp_ap1.csv", "α'/β' AP1 KCs"),
        ("kc_apbp_ap2.csv", "α'/β' AP2 KCs"),
        ("kc_g_main.csv", "γ main KCs (short-term memory)"),
        ("kc_g_dorsal.csv", "γ dorsal KCs"),
        ("kc_g_sparse.csv", "γ sparse KCs"),
    ]
    kc_total = sum(load_csv_count(filename) for filename, _ in kc_subtypes)
    kc_breakdown = ", ".join([f"{load_csv_count(f)} {name}" for f, name in kc_subtypes if load_csv_count(f) > 0])

    cell_types.append({
        "name": "Kenyon Cells (KCs)",
        "count": kc_total,
        "role": "Sparse memory encoding in mushroom body",
        "neurotransmitter": "Cholinergic",
        "connectivity": f"KC→MBON (plastic synapses modulated by DANs)",
        "category": "Core",
        "subtypes": kc_breakdown,
    })

    mbon_count = load_csv_count("mbon_all.csv")
    cell_types.append({
        "name": "Mushroom Body Output Neurons (MBONs)",
        "count": mbon_count,
        "role": "Valence decision and behavioral output",
        "neurotransmitter": "Mixed (Glutamatergic/GABAergic)",
        "connectivity": "Receives KC input, projects to motor circuits",
        "category": "Core",
    })

    dan_count = load_csv_count("dan_all.csv")
    cell_types.append({
        "name": "Dopaminergic Neurons (DANs)",
        "count": dan_count,
        "role": "Reward/punishment signal modulation",
        "neurotransmitter": "Dopamine",
        "connectivity": f"DAN→KC (plasticity gating), DAN→MBON (fast valence)",
        "category": "Core",
    })

    # Extended Components
    ln_count = load_csv_count("ln_all.csv")
    ln_gaba_count = load_csv_count("ln_gaba.csv")
    ln_chol_count = load_csv_count("ln_chol.csv")
    cell_types.append({
        "name": "Local Interneurons (LNs)",
        "count": ln_count,
        "role": "GABAergic veto gate for blocking distractor learning",
        "neurotransmitter": f"GABA ({ln_gaba_count}) / Acetylcholine ({ln_chol_count})",
        "connectivity": "Modulates PN→KC plasticity via lateral inhibition",
        "category": "Extended",
    })

    lh_count = load_csv_count("lh_all.csv")
    cell_types.append({
        "name": "Lateral Horn Neurons (LH)",
        "count": lh_count,
        "role": "Innate valence responses (hardwired odor preferences)",
        "neurotransmitter": "Mixed",
        "connectivity": "Receives PN input, bypasses learning",
        "category": "Extended",
    })

    motor_proboscis_count = load_csv_count("motor_proboscis.csv")
    motor_all_count = load_csv_count("motor_all.csv")
    cell_types.append({
        "name": "Motor Neurons (Proboscis)",
        "count": motor_proboscis_count if motor_proboscis_count > 0 else motor_all_count,
        "role": "Proboscis Extension Reflex (PER) behavioral output",
        "neurotransmitter": "Cholinergic (motor)",
        "connectivity": "Receives MBON/DN input for feeding behavior",
        "category": "Extended",
    })

    an_count = load_csv_count("an_all.csv")
    cell_types.append({
        "name": "Ascending Neurons (ANs)",
        "count": an_count,
        "role": "Sensory relay from ventral nerve cord to brain",
        "neurotransmitter": "Mixed",
        "connectivity": "VNC→Brain command signals",
        "category": "Extended",
    })

    dn_count = load_csv_count("dn_all.csv")
    cell_types.append({
        "name": "Descending Neurons (DNs)",
        "count": dn_count,
        "role": "Motor command signals from brain to VNC",
        "neurotransmitter": "Mixed",
        "connectivity": "Brain→VNC behavioral execution",
        "category": "Extended",
    })

    # NEW: CB0191 Neurons
    cb0191_count = load_csv_count("cb0191_neurons.csv")
    cell_types.append({
        "name": "CB0191 Neurons",
        "count": cb0191_count,
        "role": "Uncharacterized central processing (LAL/vest/IPS integration)",
        "neurotransmitter": "Acetylcholine (predicted)",
        "connectivity": "Postsynaptic: LAL/vest/IPS/wedge; Presynaptic: LAL/vest/IPS",
        "category": "New",
        "reference": "Schlegel et al. (2023), FBbt_20004012",
    })

    # NEW: SEZ-NSC^CAPA Neurons
    sez_count = load_csv_count("sez_nsc_capa.csv")
    cell_types.append({
        "name": "SEZ-NSC^CAPA Neurons",
        "count": sez_count,
        "role": "Nutrient-responsive hormonal regulation (post-feeding physiology)",
        "neurotransmitter": "CAPA/Pyrokinin neuropeptides",
        "connectivity": "Receives olfactory PN input; projects via NCC for systemic release",
        "category": "New",
        "reference": "Zandawala et al. (2024) eLife",
        "expected_count": 2,
    })

    total_neurons = sum(ct['count'] for ct in cell_types)

    return {
        "total_neurons": total_neurons,
        "cell_type_breakdown": cell_types,
        "generated_from": str(cache_dir),
        "flywire_version": "FAFB v783",
    }


def print_formatted_summary(summary: Dict[str, Any]) -> None:
    """Print formatted summary to console."""
    print("\n" + "="*80)
    print("PGCN OLFACTORY SYSTEM MODEL - COMPLETE CELL TYPE INVENTORY")
    print("="*80)
    print(f"\nTotal Neurons: {summary['total_neurons']:,}")
    print(f"Data Source: {summary['generated_from']}")
    print(f"FlyWire Version: {summary['flywire_version']}")
    print("\n" + "-"*80)

    # Group by category
    categories = {}
    for ct in summary['cell_type_breakdown']:
        cat = ct.get('category', 'Other')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(ct)

    # Print by category
    for category in ["Core", "Extended", "New"]:
        if category not in categories:
            continue

        print(f"\n{category} Components")
        print("-" * 40)

        for i, ct in enumerate(categories[category], 1):
            print(f"\n{i}. {ct['name']}")
            print(f"   Count: {ct['count']:,}", end="")

            # Add expected count warning for SEZ-NSC^CAPA
            if 'expected_count' in ct and ct['count'] != ct['expected_count']:
                print(f" [WARNING: Expected {ct['expected_count']}]", end="")
            print()

            print(f"   Role: {ct['role']}")
            print(f"   Neurotransmitter: {ct['neurotransmitter']}")
            print(f"   Connectivity: {ct['connectivity']}")

            if 'subtypes' in ct:
                print(f"   Subtypes: {ct['subtypes']}")

            if 'reference' in ct:
                print(f"   Reference: {ct['reference']}")

    print("\n" + "="*80)
    print(f"\n✓ Complete system: {summary['total_neurons']:,} neurons across {len(summary['cell_type_breakdown'])} cell types")
    print("✓ Integration includes 2 new cell types: CB0191 and SEZ-NSC^CAPA")
    print("\n" + "="*80)


def main(argv: list[str] | None = None) -> None:
    """Main entry point for script execution."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data") / "cache",
        help="Directory containing extracted neuron CSVs (default: data/cache)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path (default: <cache-dir>/cell_type_summary.json)",
    )

    args = parser.parse_args(argv)

    # Generate summary
    print("Generating comprehensive cell type summary...")
    summary = generate_cell_type_summary(args.cache_dir)

    # Print to console
    print_formatted_summary(summary)

    # Save JSON report
    output_path = args.output if args.output else args.cache_dir / "cell_type_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary saved to: {output_path}")


if __name__ == "__main__":
    main()
