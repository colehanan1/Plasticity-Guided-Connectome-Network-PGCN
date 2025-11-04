#!/usr/bin/env python3
"""
FlyWire Classification Data Analyzer

This script analyzes the FlyWire classification.csv.gz file to extract comprehensive
information about available cell types, neurotransmitters, and brain regions.
Outputs detailed summaries in a format optimized for understanding what neural
components are available for extraction.

Usage: python analyze_flywire_data.py --data-dir /path/to/flywire/data
"""

import pandas as pd
import numpy as np
import gzip
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import json

def load_flywire_files(data_dir):
    """Load all relevant FlyWire CSV files"""
    data_dir = Path(data_dir)
    
    files = {}
    
    # Classification file (hierarchical annotations)
    classification_file = data_dir / "classification.csv.gz"
    if classification_file.exists():
        print(f"Loading classification from {classification_file}")
        files['classification'] = pd.read_csv(classification_file)
        print(f"  ✓ Loaded {len(files['classification'])} neurons")
    
    # Cell types file (primary annotations)  
    cell_types_file = data_dir / "consolidated_cell_types.csv.gz"
    if cell_types_file.exists():
        print(f"Loading cell types from {cell_types_file}")
        files['cell_types'] = pd.read_csv(cell_types_file)
        print(f"  ✓ Loaded {len(files['cell_types'])} neurons")
    
    # Neurotransmitter predictions
    neurons_file = data_dir / "neurons.csv.gz"
    if neurons_file.exists():
        print(f"Loading neurotransmitters from {neurons_file}")
        files['neurons'] = pd.read_csv(neurons_file)
        print(f"  ✓ Loaded {len(files['neurons'])} neurons")
    
    # Names/regions file
    names_file = data_dir / "names.csv.gz"
    if names_file.exists():
        print(f"Loading names/regions from {names_file}")
        files['names'] = pd.read_csv(names_file)
        print(f"  ✓ Loaded {len(files['names'])} neurons")
    
    # Processed labels (glomerulus assignments, etc.)
    labels_file = data_dir / "processed_labels.csv.gz"
    if labels_file.exists():
        print(f"Loading processed labels from {labels_file}")
        files['processed_labels'] = pd.read_csv(labels_file)
        print(f"  ✓ Loaded {len(files['processed_labels'])} neurons")
    
    return files

def analyze_classification_hierarchy(df):
    """Analyze the hierarchical classification structure"""
    
    analysis = {
        'summary': {
            'total_neurons': len(df),
            'neurons_with_flow': df['flow'].notna().sum(),
            'neurons_with_super_class': df['super_class'].notna().sum(),
            'neurons_with_class': df['class'].notna().sum(),
            'neurons_with_sub_class': df['sub_class'].notna().sum(),
            'neurons_with_hemilineage': df['hemilineage'].notna().sum(),
            'neurons_with_side': df['side'].notna().sum(),
            'neurons_with_nerve': df['nerve'].notna().sum(),
        }
    }
    
    # Unique values for each column
    print("\n" + "="*80)
    print("HIERARCHICAL CLASSIFICATION ANALYSIS")
    print("="*80)
    
    for col in ['flow', 'super_class', 'class', 'sub_class', 'side', 'nerve']:
        unique_vals = df[col].dropna().unique()
        analysis[f'{col}_unique_values'] = sorted(unique_vals.tolist())
        
        print(f"\n{col.upper()} ({len(unique_vals)} unique values):")
        print("-" * (len(col) + 25))
        
        # Show value counts
        value_counts = df[col].value_counts(dropna=False)
        for val, count in value_counts.head(20).items():  # Top 20
            if pd.isna(val):
                print(f"  [EMPTY]: {count:,} neurons")
            else:
                print(f"  {val}: {count:,} neurons")
        
        if len(value_counts) > 20:
            print(f"  ... and {len(value_counts) - 20} more values")
    
    # Cross-tabulation of important combinations
    print(f"\n" + "="*50)
    print("KEY COMBINATIONS FOR NEURAL COMPONENT IDENTIFICATION")
    print("="*50)
    
    # Super_class vs class combinations for target components
    target_terms = [
        'olfactory', 'visual', 'motor', 'interneuron', 'local', 
        'ascending', 'descending', 'projection', 'sensory'
    ]
    
    found_combinations = []
    
    for term in target_terms:
        # Check super_class
        super_matches = df[df['super_class'].str.contains(term, case=False, na=False)]
        if len(super_matches) > 0:
            found_combinations.append({
                'search_term': term,
                'field': 'super_class',
                'count': len(super_matches),
                'example_classes': super_matches['class'].value_counts().head(5).to_dict()
            })
        
        # Check class
        class_matches = df[df['class'].str.contains(term, case=False, na=False)]
        if len(class_matches) > 0:
            found_combinations.append({
                'search_term': term,
                'field': 'class', 
                'count': len(class_matches),
                'example_sub_classes': class_matches['sub_class'].value_counts().head(5).to_dict()
            })
        
        # Check sub_class
        sub_matches = df[df['sub_class'].str.contains(term, case=False, na=False)]
        if len(sub_matches) > 0:
            found_combinations.append({
                'search_term': term,
                'field': 'sub_class',
                'count': len(sub_matches),
                'example_values': sub_matches['sub_class'].value_counts().head(5).to_dict()
            })
    
    analysis['target_component_matches'] = found_combinations
    
    for match in found_combinations:
        print(f"\nFound '{match['search_term']}' in {match['field']}: {match['count']:,} neurons")
        if 'example_classes' in match:
            print("  Top classes:")
            for cls, count in match['example_classes'].items():
                print(f"    {cls}: {count}")
        elif 'example_sub_classes' in match:
            print("  Top sub_classes:")
            for sub, count in match['example_sub_classes'].items():
                print(f"    {sub}: {count}")
        elif 'example_values' in match:
            print("  Top values:")
            for val, count in match['example_values'].items():
                print(f"    {val}: {count}")
    
    return analysis

def analyze_neurotransmitters(df):
    """Analyze neurotransmitter predictions"""
    if 'nt_type' not in df.columns:
        print("No neurotransmitter data found")
        return {}
    
    print(f"\n" + "="*50)
    print("NEUROTRANSMITTER ANALYSIS")
    print("="*50)
    
    nt_counts = df['nt_type'].value_counts(dropna=False)
    
    analysis = {
        'total_predictions': df['nt_type'].notna().sum(),
        'nt_types': nt_counts.to_dict()
    }
    
    print(f"Neurons with neurotransmitter predictions: {analysis['total_predictions']:,}")
    print("\nNeurotransmitter distribution:")
    for nt, count in nt_counts.items():
        if pd.isna(nt):
            print(f"  [UNKNOWN]: {count:,} neurons")
        else:
            print(f"  {nt}: {count:,} neurons")
    
    return analysis

def analyze_brain_regions(df):
    """Analyze brain region annotations"""
    region_columns = ['input_neuropils', 'output_neuropils']
    
    analysis = {}
    
    for col in region_columns:
        if col not in df.columns:
            continue
            
        print(f"\n" + "="*50)
        print(f"{col.upper()} ANALYSIS")
        print("="*50)
        
        # Parse neuropil lists (they're often comma-separated)
        all_regions = []
        for regions_str in df[col].dropna():
            if isinstance(regions_str, str):
                regions = [r.strip() for r in regions_str.split(',')]
                all_regions.extend(regions)
        
        region_counts = Counter(all_regions)
        analysis[col] = dict(region_counts.most_common(30))  # Top 30 regions
        
        print(f"Neurons with {col}: {df[col].notna().sum():,}")
        print(f"Unique regions found: {len(region_counts)}")
        print(f"\nTop regions:")
        
        for region, count in region_counts.most_common(20):
            print(f"  {region}: {count:,} neurons")
    
    return analysis

def find_target_neural_components(files):
    """Search for target neural components across all files"""
    
    print(f"\n" + "="*80)
    print("SEARCHING FOR TARGET NEURAL COMPONENTS")
    print("="*80)
    
    # Target components we're looking for
    targets = {
        'gaba_interneurons': {
            'description': 'GABAergic Local Interneurons in Antennal Lobe',
            'search_terms': ['GABA', 'interneuron', 'local'],
            'region_terms': ['AL', 'antennal']
        },
        'chol_interneurons': {
            'description': 'Cholinergic Local Interneurons in Antennal Lobe', 
            'search_terms': ['ACH', 'interneuron', 'local', 'cholinergic'],
            'region_terms': ['AL', 'antennal']
        },
        'lateral_horn': {
            'description': 'Lateral Horn Neurons',
            'search_terms': ['lateral', 'horn', 'LH'],
            'region_terms': ['LH', 'lateral']
        },
        'motor_proboscis': {
            'description': 'Proboscis Motor Neurons',
            'search_terms': ['motor', 'MN', 'proboscis'],
            'region_terms': ['GNG', 'gnathal']
        },
        'ascending': {
            'description': 'Ascending Neurons (VNC→Brain)',
            'search_terms': ['ascending', 'AN'],
            'region_terms': ['VNC', 'ventral']
        },
        'descending': {
            'description': 'Descending Neurons (Brain→VNC)',
            'search_terms': ['descending', 'DN'],
            'region_terms': ['VNC', 'ventral']
        }
    }
    
    results = {}
    
    for target_name, target_info in targets.items():
        print(f"\n{'-' * 60}")
        print(f"SEARCHING FOR: {target_info['description']}")
        print(f"{'-' * 60}")
        
        target_results = {
            'description': target_info['description'],
            'found_in_files': {},
            'candidate_neurons': []
        }
        
        # Search across all loaded files
        for file_name, df in files.items():
            file_matches = []
            
            # Search in text columns
            text_columns = df.select_dtypes(include=[object]).columns
            
            for col in text_columns:
                for search_term in target_info['search_terms']:
                    matches = df[df[col].str.contains(search_term, case=False, na=False)]
                    if len(matches) > 0:
                        file_matches.append({
                            'column': col,
                            'search_term': search_term,
                            'count': len(matches),
                            'sample_values': matches[col].value_counts().head(3).to_dict()
                        })
            
            if file_matches:
                target_results['found_in_files'][file_name] = file_matches
                print(f"  Found in {file_name}:")
                for match in file_matches:
                    print(f"    {match['column']} + '{match['search_term']}': {match['count']} matches")
                    for val, count in match['sample_values'].items():
                        print(f"      '{val}': {count}")
        
        results[target_name] = target_results
    
    return results

def generate_codex_queries(analysis_results):
    """Generate Codex queries based on discovered terminology"""
    
    print(f"\n" + "="*80)
    print("RECOMMENDED CODEX QUERIES")
    print("="*80)
    
    queries = {}
    
    # Extract discovered terminology
    classification = analysis_results.get('classification', {})
    
    if 'class_unique_values' in classification:
        classes = classification['class_unique_values']
        
        # Generate queries for different components
        
        # 1. Interneurons
        interneuron_classes = [c for c in classes if 'intern' in c.lower()]
        if interneuron_classes:
            queries['interneurons'] = []
            for cls in interneuron_classes:
                queries['interneurons'].append(f"class == {cls}")
        
        # 2. Motor neurons
        motor_classes = [c for c in classes if any(term in c.lower() for term in ['motor', 'mn', 'muscle'])]
        if motor_classes:
            queries['motor_neurons'] = []
            for cls in motor_classes:
                queries['motor_neurons'].append(f"class == {cls}")
        
        # 3. Ascending neurons
        ascending_classes = [c for c in classes if 'ascend' in c.lower() or c.upper() == 'AN']
        if ascending_classes:
            queries['ascending_neurons'] = []
            for cls in ascending_classes:
                queries['ascending_neurons'].append(f"class == {cls}")
        
        # 4. Descending neurons  
        descending_classes = [c for c in classes if 'descend' in c.lower() or c.upper() == 'DN']
        if descending_classes:
            queries['descending_neurons'] = []
            for cls in descending_classes:
                queries['descending_neurons'].append(f"class == {cls}")
    
    # Print recommended queries
    for component, component_queries in queries.items():
        print(f"\n{component.upper()}:")
        for query in component_queries[:5]:  # Top 5 queries
            print(f"  {query}")
    
    return queries

def main():
    parser = argparse.ArgumentParser(description='Analyze FlyWire classification data')
    parser.add_argument('--data-dir', required=True, help='Directory containing FlyWire CSV files')
    parser.add_argument('--output', default='flywire_analysis.txt', help='Output file for analysis')
    
    args = parser.parse_args()
    
    # Load all FlyWire files
    files = load_flywire_files(args.data_dir)
    
    if not files:
        print("ERROR: No FlyWire files found!")
        return
    
    # Comprehensive analysis
    analysis_results = {}
    
    # 1. Hierarchical classification analysis
    if 'classification' in files:
        analysis_results['classification'] = analyze_classification_hierarchy(files['classification'])
    
    # 2. Neurotransmitter analysis
    if 'neurons' in files:
        analysis_results['neurotransmitters'] = analyze_neurotransmitters(files['neurons'])
    
    # 3. Brain region analysis
    for file_name, df in files.items():
        if any(col in df.columns for col in ['input_neuropils', 'output_neuropils']):
            analysis_results[f'{file_name}_regions'] = analyze_brain_regions(df)
    
    # 4. Target component search
    analysis_results['target_components'] = find_target_neural_components(files)
    
    # 5. Generate Codex queries
    analysis_results['recommended_queries'] = generate_codex_queries(analysis_results)
    
    # 6. Summary for extraction code
    print(f"\n" + "="*80)
    print("SUMMARY FOR PGCN EXTRACTION CODE")
    print("="*80)
    
    print("\nTo extract neural components in your existing pipeline, use these patterns:")
    
    if 'classification' in analysis_results:
        class_values = analysis_results['classification'].get('class_unique_values', [])
        super_class_values = analysis_results['classification'].get('super_class_unique_values', [])
        
        print(f"\nAvailable classes ({len(class_values)}):")
        for cls in class_values[:10]:
            print(f"  '{cls}'")
        
        print(f"\nAvailable super_classes ({len(super_class_values)}):")
        for sup in super_class_values:
            print(f"  '{sup}'")
    
    # Save detailed results to file
    output_file = Path(args.output)
    with open(output_file, 'w') as f:
        f.write("FlyWire Classification Data Analysis\n")
        f.write("="*50 + "\n\n")
        
        # Write JSON summary
        f.write("ANALYSIS RESULTS (JSON):\n")
        f.write("-" * 30 + "\n")
        f.write(json.dumps(analysis_results, indent=2, default=str))
    
    print(f"\nDetailed analysis saved to: {output_file}")
    print(f"\nNext steps:")
    print(f"  1. Review the analysis above")
    print(f"  2. Test the recommended Codex queries")
    print(f"  3. Use the discovered terminology in your extraction scripts")

if __name__ == "__main__":
    main()