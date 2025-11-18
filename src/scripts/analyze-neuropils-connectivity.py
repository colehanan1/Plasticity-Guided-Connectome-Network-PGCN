#!/usr/bin/env python3
"""
FlyWire Neuropil and Connectivity Analyzer

This script analyzes your FlyWire connections and neuropil data to discover:
1. Exact calyx region names (for PN→KC connectivity)
2. Mushroom body region names 
3. Connection patterns between ALPNs and Kenyon cells
4. Output regions for each neuron type

Run this to get the perfect data for fixing your PN→KC connectivity issue.

Usage: python analyze_neuropils_connectivity.py --data-dir data/flywire
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from collections import Counter

def analyze_neuropil_patterns(connections_df, output_file="neuropil_analysis.txt"):
    """Analyze all neuropil patterns in the connections data"""
    
    print("="*80)
    print("NEUROPIL & CONNECTIVITY ANALYSIS")
    print("="*80)
    
    results = []
    
    # 1. Basic connection stats
    results.append(f"\n=== CONNECTION DATA OVERVIEW ===")
    results.append(f"Total connections: {len(connections_df):,}")
    results.append(f"Connection columns: {list(connections_df.columns)}")
    
    # 2. Neuropil analysis
    if 'neuropil' in connections_df.columns:
        neuropil_counts = connections_df['neuropil'].value_counts()
        results.append(f"\n=== ALL NEUROPILS ({len(neuropil_counts)} unique) ===")
        
        # Top 30 neuropils
        results.append("Top 30 neuropils by connection count:")
        for neuropil, count in neuropil_counts.head(30).items():
            results.append(f"  {neuropil}: {count:,} connections")
        
        # Search for key regions
        key_terms = {
            'CALYX': ['CA', 'calyx', 'CALYX'],
            'MUSHROOM_BODY': ['MB', 'mushroom', 'body'],
            'ANTENNAL_LOBE': ['AL', 'antennal', 'lobe'], 
            'LATERAL_HORN': ['LH', 'lateral', 'horn'],
            'VNC': ['VNC', 'ventral', 'nerve'],
            'GNATHAL': ['GNG', 'gnathal']
        }
        
        results.append(f"\n=== KEY REGION SEARCH ===")
        for region_type, search_terms in key_terms.items():
            results.append(f"\n{region_type}:")
            found_regions = set()
            
            for term in search_terms:
                matches = connections_df[connections_df['neuropil'].str.contains(term, case=False, na=False)]
                if len(matches) > 0:
                    unique_regions = matches['neuropil'].unique()
                    found_regions.update(unique_regions)
                    results.append(f"  Search '{term}': {len(matches):,} connections")
            
            if found_regions:
                results.append("  Found regions:")
                for region in sorted(found_regions)[:10]:  # Top 10
                    count = neuropil_counts.get(region, 0)
                    results.append(f"    {region}: {count:,}")
            else:
                results.append("  No regions found")
    
    else:
        results.append("ERROR: No 'neuropil' column found in connections!")
        results.append(f"Available columns: {list(connections_df.columns)}")
    
    # Save results
    with open(output_file, 'w') as f:
        f.write('\n'.join(results))
    
    # Print to console too
    for line in results:
        print(line)
    
    return results

def analyze_alpn_kc_connectivity(connections_df, alpn_ids, kc_ids):
    """Analyze connectivity patterns between ALPNs and KCs"""
    
    print(f"\n=== ALPN→KC CONNECTIVITY ANALYSIS ===")
    
    # Filter connections to ALPN→KC
    alpn_to_kc = connections_df[
        (connections_df['pre_root_id'].isin(alpn_ids)) & 
        (connections_df['post_root_id'].isin(kc_ids))
    ]
    
    print(f"Total ALPN→KC connections: {len(alpn_to_kc):,}")
    
    if len(alpn_to_kc) > 0:
        # Analyze by neuropil
        if 'neuropil' in alpn_to_kc.columns:
            neuropil_counts = alpn_to_kc['neuropil'].value_counts()
            print(f"ALPN→KC connections by neuropil:")
            for neuropil, count in neuropil_counts.items():
                print(f"  {neuropil}: {count:,} connections")
        
        # Synapse strength analysis
        if 'synapse_count' in alpn_to_kc.columns:
            print(f"\nSynapse strength distribution:")
            print(f"  Mean synapses per connection: {alpn_to_kc['synapse_count'].mean():.1f}")
            print(f"  Min synapses: {alpn_to_kc['synapse_count'].min()}")
            print(f"  Max synapses: {alpn_to_kc['synapse_count'].max()}")
            
            # Count by threshold
            for threshold in [1, 2, 5, 10]:
                above_thresh = len(alpn_to_kc[alpn_to_kc['synapse_count'] >= threshold])
                print(f"  Connections ≥{threshold} synapses: {above_thresh:,}")
        
        return alpn_to_kc
    
    else:
        print("WARNING: No direct ALPN→KC connections found!")
        
        # Debug: check if ALPNs and KCs exist separately
        alpn_as_pre = connections_df[connections_df['pre_root_id'].isin(alpn_ids)]
        kc_as_post = connections_df[connections_df['post_root_id'].isin(kc_ids)]
        
        print(f"  ALPNs as presynaptic: {len(alpn_as_pre):,} connections")
        print(f"  KCs as postsynaptic: {len(kc_as_post):,} connections")
        
        if len(alpn_as_pre) > 0:
            print("  ALPN output regions:")
            alpn_regions = alpn_as_pre['neuropil'].value_counts().head(10)
            for region, count in alpn_regions.items():
                print(f"    {region}: {count:,}")
        
        if len(kc_as_post) > 0:
            print("  KC input regions:")
            kc_regions = kc_as_post['neuropil'].value_counts().head(10)
            for region, count in kc_regions.items():
                print(f"    {region}: {count:,}")
    
    return pd.DataFrame()

def load_extracted_neuron_ids(cache_dir):
    """Load neuron IDs from your extracted CSV files"""
    cache_dir = Path(cache_dir)
    
    neuron_ids = {}
    
    # Load ALPN IDs
    alpn_file = cache_dir / "alpn_extracted.csv"
    if alpn_file.exists():
        alpn_df = pd.read_csv(alpn_file)
        neuron_ids['alpn'] = alpn_df['root_id'].unique()
        print(f"Loaded ALPN IDs: {len(neuron_ids['alpn'])}")
    
    # You could also load from other files if needed
    # kc_file = cache_dir / "kc_all.csv"  # etc.
    
    return neuron_ids

def main():
    parser = argparse.ArgumentParser(description='Analyze FlyWire neuropils and connectivity')
    parser.add_argument('--data-dir', required=True, help='Directory with FlyWire CSV files')
    parser.add_argument('--cache-dir', default='data/cache', help='Directory with extracted neuron CSVs')
    parser.add_argument('--output', default='neuropil_connectivity_analysis.txt', help='Output analysis file')
    
    args = parser.parse_args()
    
    # Load connections data
    print("Loading FlyWire connections...")
    connections_file = Path(args.data_dir) / "connections_princeton.csv.gz"
    
    if not connections_file.exists():
        print(f"ERROR: Connections file not found at {connections_file}")
        return
    
    connections_df = pd.read_csv(connections_file)
    print(f"✓ Loaded {len(connections_df):,} connections")
    
    # Analyze neuropil patterns
    results = analyze_neuropil_patterns(connections_df, args.output)
    
    # Load extracted neuron IDs and analyze connectivity
    neuron_ids = load_extracted_neuron_ids(args.cache_dir)
    
    if 'alpn' in neuron_ids:
        # We need KC IDs - let's get them from classification
        classification_file = Path(args.data_dir) / "classification.csv.gz"
        if classification_file.exists():
            classification_df = pd.read_csv(classification_file)
            kc_df = classification_df[classification_df['class'] == 'Kenyon_Cell']
            kc_ids = kc_df['root_id'].unique()
            print(f"Found {len(kc_ids)} Kenyon cells from classification")
            
            # Analyze ALPN→KC connectivity patterns
            alpn_kc_connections = analyze_alpn_kc_connectivity(
                connections_df, neuron_ids['alpn'], kc_ids
            )
        else:
            print("WARNING: No classification file found - skipping ALPN→KC analysis")
    
    print(f"\nDetailed analysis saved to: {args.output}")
    
    # Generate Codex queries for specific findings
    print(f"\n=== RECOMMENDED CODEX QUERIES FOR FINE-TUNING ===")
    print("Based on analysis, try these specific queries:")
    
    if 'neuropil' in connections_df.columns:
        # Find top calyx regions
        ca_connections = connections_df[connections_df['neuropil'].str.contains('CA', case=False, na=False)]
        if len(ca_connections) > 0:
            top_ca_regions = ca_connections['neuropil'].value_counts().head(5)
            print("\nTop CA regions (use these for calyx filtering):")
            for region, count in top_ca_regions.items():
                print(f"  output_neuropils >> {region}")
                print(f"    [Click to test](https://codex.flywire.ai/app/search?dataset=fafb&filter_string=output_neuropils+%3E%3E+{region})")
        
        # Find ALPN output regions specifically
        alpn_file = Path(args.cache_dir) / "alpn_extracted.csv"
        if alpn_file.exists():
            alpn_df = pd.read_csv(alpn_file)
            alpn_connections = connections_df[connections_df['pre_root_id'].isin(alpn_df['root_id'])]
            if len(alpn_connections) > 0:
                alpn_regions = alpn_connections['neuropil'].value_counts().head(10)
                print(f"\nTop ALPN output regions:")
                for region, count in alpn_regions.items():
                    print(f"  {region}: {count:,} ALPN connections")

if __name__ == "__main__":
    main()