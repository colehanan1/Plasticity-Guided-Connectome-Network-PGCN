#!/usr/bin/env python3
"""
Extract FlyWire root IDs from PGCN cache and format for Codex copy-paste.

Usage:
    python scripts/extract_neuron_ids_for_codex.py --cache-dir data/cache --output-dir reports

Outputs:
    - pgcn_alpn_ids.txt          : All 353 olfactory ALPNs (copy-paste into Codex)
    - pgcn_kc_ids.txt            : All 5,177 Kenyon cells (copy-paste into Codex)
    - pgcn_mbon_ids.txt          : All 96 MBONs (copy-paste into Codex)
    - pgcn_dan_ids.txt           : All 584 DANs (copy-paste into Codex)
    - pgcn_dan_mb_ids.txt        : Only 285 MB-targeting DANs (copy-paste into Codex)
    - pgcn_all_ids.txt           : All neurons at once (NOT RECOMMENDED - too many)
    - pgcn_circuit_exact_ids.txt : Exact circuit subset (353 PN + 5177 KC + 96 MBON + 285 DAN)

Each .txt file contains comma-separated root IDs ready to paste into FlyWire search.
"""

import argparse
import csv
import pandas as pd
from pathlib import Path
from typing import List, Tuple


def extract_root_ids_from_csv(csv_path: Path, root_id_column: str = "root_id") -> List[str]:
    """
    Extract root IDs from a CSV file.
    
    Args:
        csv_path: Path to CSV file (supports .gz compression)
        root_id_column: Name of the column containing root IDs
        
    Returns:
        List of root ID strings (not integers, to preserve precision)
    """
    try:
        df = pd.read_csv(csv_path, dtype={root_id_column: str})
        root_ids = df[root_id_column].dropna().tolist()
        # Ensure they're strings, strip whitespace
        root_ids = [str(int(rid)).strip() if str(rid).strip() else None for rid in root_ids]
        root_ids = [rid for rid in root_ids if rid]  # Remove None
        return root_ids
    except Exception as e:
        print(f"ERROR reading {csv_path}: {e}")
        return []


def write_ids_to_codex_format(root_ids: List[str], output_path: Path) -> int:
    """
    Write root IDs in Codex-compatible copy-paste format (comma-separated, single line).
    
    Args:
        root_ids: List of root ID strings
        output_path: Path to output .txt file
        
    Returns:
        Number of IDs written
    """
    if not root_ids:
        print(f"WARNING: No IDs to write to {output_path}")
        return 0
    
    # Format as comma-separated, single line
    formatted = ", ".join(root_ids)
    
    with open(output_path, "w") as f:
        f.write(formatted)
    
    print(f"✓ Written {len(root_ids)} IDs to {output_path}")
    return len(root_ids)


def main():
    parser = argparse.ArgumentParser(
        description="Extract FlyWire root IDs from PGCN cache and format for Codex"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache"),
        help="Path to PGCN cache directory (default: data/cache)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Path to output directory (default: reports)"
    )
    
    args = parser.parse_args()
    
    cache_dir = args.cache_dir
    output_dir = args.output_dir
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading from cache: {cache_dir}")
    print(f"Writing to: {output_dir}\n")
    
    # Dictionary of files to extract
    # Maps: (csv_filename, output_name) -> description
    extractions = {
        ("alpn_extracted.csv", "pgcn_alpn_ids.txt"): "Olfactory PNs (353)",
        ("mbon_all.csv", "pgcn_mbon_ids.txt"): "All MBONs (96)",
        ("dan_all.csv", "pgcn_dan_ids.txt"): "All DANs (584)",
        ("dan_mb.csv", "pgcn_dan_mb_ids.txt"): "MB-targeting DANs only (285)",
    }
    
    # Extract KC IDs from all KC subtype files
    kc_files = [
        "kc_ab.csv",
        "kc_abp.csv",
        "kc_g_main.csv",
        "kc_g_dorsal.csv",
        "kc_g_sparse.csv",
        "kc_apbp_main.csv",
        "kc_apbp_ap1.csv",
        "kc_apbp_ap2.csv",
    ]
    
    results = {}
    
    # Process each extraction
    for (csv_name, output_name), description in extractions.items():
        csv_path = cache_dir / csv_name
        output_path = output_dir / output_name
        
        if csv_path.exists():
            root_ids = extract_root_ids_from_csv(csv_path)
            count = write_ids_to_codex_format(root_ids, output_path)
            results[output_name] = (count, description)
        else:
            print(f"⚠ Missing: {csv_path}")
    
    # Combine all KC files
    all_kc_ids = []
    for kc_file in kc_files:
        csv_path = cache_dir / kc_file
        if csv_path.exists():
            kc_ids = extract_root_ids_from_csv(csv_path)
            all_kc_ids.extend(kc_ids)
        else:
            print(f"⚠ Missing KC subtype file: {csv_path}")
    
    if all_kc_ids:
        # Remove duplicates while preserving order
        all_kc_ids = list(dict.fromkeys(all_kc_ids))
        output_path = output_dir / "pgcn_kc_ids.txt"
        count = write_ids_to_codex_format(all_kc_ids, output_path)
        results["pgcn_kc_ids.txt"] = (count, f"All Kenyon cells ({count})")
    
    # Create combined files for different use cases
    print("\n--- Creating combined files ---\n")
    
    # Option 1: Exact circuit (353 PN + 5177 KC + 96 MBON + 285 DAN MB-only)
    exact_circuit_ids = []
    for file_name in ["pgcn_alpn_ids.txt", "pgcn_kc_ids.txt", "pgcn_mbon_ids.txt", "pgcn_dan_mb_ids.txt"]:
        file_path = output_dir / file_name
        if file_path.exists():
            with open(file_path, "r") as f:
                ids = [id.strip() for id in f.read().split(",")]
                exact_circuit_ids.extend([id for id in ids if id])
    
    if exact_circuit_ids:
        # Remove duplicates
        exact_circuit_ids = list(dict.fromkeys(exact_circuit_ids))
        output_path = output_dir / "pgcn_circuit_exact_ids.txt"
        count = write_ids_to_codex_format(exact_circuit_ids, output_path)
        results["pgcn_circuit_exact_ids.txt"] = (count, "EXACT CIRCUIT (353 PN + 5177 KC + 96 MBON + 285 DAN MB)")
    
    # Option 2: All neurons (353 PN + 5177 KC + 96 MBON + 584 all DAN)
    all_ids = []
    for file_name in ["pgcn_alpn_ids.txt", "pgcn_kc_ids.txt", "pgcn_mbon_ids.txt", "pgcn_dan_ids.txt"]:
        file_path = output_dir / file_name
        if file_path.exists():
            with open(file_path, "r") as f:
                ids = [id.strip() for id in f.read().split(",")]
                all_ids.extend([id for id in ids if id])
    
    if all_ids:
        # Remove duplicates
        all_ids = list(dict.fromkeys(all_ids))
        output_path = output_dir / "pgcn_all_ids.txt"
        count = write_ids_to_codex_format(all_ids, output_path)
        results["pgcn_all_ids.txt"] = (count, "ALL NEURONS (353 PN + 5177 KC + 96 MBON + 584 DAN) - NOT RECOMMENDED")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: Root ID Files Ready for Codex Copy-Paste")
    print("=" * 80 + "\n")
    
    for file_name, (count, description) in sorted(results.items()):
        print(f"✓ {file_name:35} : {count:6} neurons : {description}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDED LOADING ORDER (Load layer-by-layer for best performance)")
    print("=" * 80 + "\n")
    
    print("1. Load OLFACTORY PNs:")
    print(f"   File: {output_dir}/pgcn_alpn_ids.txt")
    print("   Copy contents → FlyWire Search → Press Enter → Color: BLUE\n")
    
    print("2. Load KENYON CELLS:")
    print(f"   File: {output_dir}/pgcn_kc_ids.txt")
    print("   Copy contents → FlyWire Search → Press Enter → Color: GREEN\n")
    
    print("3. Load MBONs:")
    print(f"   File: {output_dir}/pgcn_mbon_ids.txt")
    print("   Copy contents → FlyWire Search → Press Enter → Color: YELLOW\n")
    
    print("4. Load MB-TARGETING DANs (RECOMMENDED for olfactory circuit):")
    print(f"   File: {output_dir}/pgcn_dan_mb_ids.txt")
    print("   Copy contents → FlyWire Search → Press Enter → Color: RED\n")
    
    print("-" * 80)
    print("\nALTERNATIVE: Load exact circuit all at once:")
    print(f"   File: {output_dir}/pgcn_circuit_exact_ids.txt")
    print("   (353 PN + 5177 KC + 96 MBON + 285 DAN MB-only)\n")
    
    print("WARNING: Do NOT use pgcn_all_ids.txt unless you have a high-performance GPU/browser.")
    print("It contains 6,210 neurons and will be very slow to render.\n")
    
    print("=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("\n1. Open FlyWire: https://flywire.ai/")
    print("2. Click Search (magnifying glass icon)")
    print("3. Open one of the .txt files generated above")
    print("4. Copy all the text (Ctrl+A → Ctrl+C)")
    print("5. Paste into FlyWire search bar → Press Enter")
    print("6. Wait ~10 seconds for neurons to render")
    print("7. Right-click result → 'Change color' → Select color (Blue/Green/Yellow/Red)")
    print("8. Repeat steps 2-7 for each layer")
    print("9. Press 'M' to toggle brain mesh")
    print("10. Press 'Ctrl+P' to export as PNG or MP4\n")


if __name__ == "__main__":
    main()
